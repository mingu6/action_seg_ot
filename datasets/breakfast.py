import os
import os.path as path

import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, root_dir: str, n_frames, standardise=True, split: str = None, random=True, n_videos=None, action_class='all'):
        self.root_dir = path.join(root_dir, 'Breakfast')
        self.video_fnames = sorted([fname for fname in os.listdir(path.join(self.root_dir, 'groundTruth'))
                                    if len(fname.split('_')) == 4])
        if action_class != ['all']:
            if type(action_class) is list:
                self.video_fnames = [fname for fname in self.video_fnames if fname.split('_')[-1] in action_class]
            else:
                self.video_fnames = [fname for fname in self.video_fnames if fname.split('_')[-1] == action_class]
        if n_videos is not None:
            # inds = np.random.permutation(len(self.video_fnames))[:n_videos]
            # self.video_fnames = sorted([self.video_fnames[ind] for ind in inds])
            self.video_fnames = self.video_fnames[::int(len(self.video_fnames) / n_videos)]
        def prep(x):
            i, nm = x.rstrip().split(' ')
            return nm, int(i)
        action_mapping = list(map(prep, open(path.join(self.root_dir, 'mapping/mapping.txt'))))
        self.action_mapping = dict(action_mapping)
        self.n_subactions = len(set(self.action_mapping.keys()))
        self.n_frames = n_frames
        self.standardise = standardise
        self.random = random

    def __len__(self):
        return len(self.video_fnames)
    
    def __getitem__(self, idx):
        video_fname = self.video_fnames[idx]
        gt = [line.rstrip() for line in open(path.join(self.root_dir, 'groundTruth', video_fname))]
        inds, mask = self._partition_and_sample(self.n_frames, len(gt))
        gt = torch.Tensor([self.action_mapping[gt[ind]] for ind in inds])
        person, cam_name, _, action = video_fname.split('_')
        feat_fname = path.join(self.root_dir, 'features/s1', action, video_fname + '.txt')
        features = np.loadtxt(feat_fname)[inds, :]
        if self.standardise:  # normalize features
            zmask = np.ones(features.shape[0], dtype=bool)
            for rdx, row in enumerate(features):
                if np.sum(row) == 0:
                    zmask[rdx] = False
            z = features[zmask] - np.mean(features[zmask], axis=0)
            z = z / np.std(features[zmask], axis=0)
            features = np.zeros(features.shape)
            features[zmask] = z
            features = np.nan_to_num(features)
            features /= np.sqrt(features.shape[1])
        # mask = torch.from_numpy(mask * zmask)
        features = torch.from_numpy(features).float()
        return features, mask, gt, action, person, cam_name, gt.unique().shape[0]
    
    def _partition_and_sample(self, n_samples, n_frames):
        if n_samples is None:
            indices = np.arange(n_frames)
            mask = np.full(n_frames, 1, dtype=bool)
        elif n_samples < n_frames:
            if self.random:
                boundaries = np.linspace(0, n_frames-1, n_samples+1).astype(int)
                indices = np.random.randint(low=boundaries[:-1], high=boundaries[1:])
            else:
                indices = np.linspace(0, n_frames-1, n_samples).astype(int)
            mask = np.full(n_samples, 1, dtype=bool)
        else:
            indices = np.concatenate((np.arange(n_frames), np.full(n_samples - n_frames, n_frames - 1)))
            mask = np.concatenate((np.full(n_frames, 1, dtype=bool), np.zeros(n_samples - n_frames, dtype=bool)))
        return indices, mask
