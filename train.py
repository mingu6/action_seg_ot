import argparse
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

from datasets.video_dataset import VideoDataset
import asot
from utils import *
from metrics import ClusteringMetrics, indep_eval_metrics

num_eps = 1e-11


class VideoSSL(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=1e-4, layer_sizes=[64, 128, 40], n_clusters=20, alpha_train=0.3, alpha_eval=0.3, n_ot_train=[25, 10], n_ot_eval=[25, 10],
                 train_eps=0.06, eval_eps=0.01, ub_proj_type='kl', ub_train=0.05, ub_eval=0.01, temp=0.1, radius_gw=0.04, learn_clusters=True,
                 n_frames=256, rho=0.1, exclude_cls=None, visualize=False):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_clusters = n_clusters
        self.learn_clusters = learn_clusters
        self.layer_sizes = layer_sizes
        self.exclude_cls = exclude_cls
        self.visualize = visualize

        self.alpha_train = alpha_train
        self.alpha_eval = alpha_eval
        self.n_ot_train = n_ot_train
        self.n_ot_eval = n_ot_eval
        self.train_eps = train_eps
        self.eval_eps = eval_eps
        self.radius_gw = radius_gw
        self.ub_proj_type = ub_proj_type
        self.ub_train = ub_train
        self.ub_eval = ub_eval

        self.temp = temp
        self.n_frames = n_frames
        self.rho = rho

        # initialize MLP
        layers = [nn.Sequential(nn.Linear(sz, sz1), nn.ReLU()) for sz, sz1 in zip(layer_sizes[:-2], layer_sizes[1:-1])]
        layers += [nn.Linear(layer_sizes[-2], layer_sizes[-1])]
        self.mlp = nn.Sequential(*layers)

        # initialize cluster centers/codebook
        d = self.layer_sizes[-1]
        self.clusters = nn.parameter.Parameter(data=F.normalize(torch.randn(self.n_clusters, d), dim=-1), requires_grad=learn_clusters)

        # initialize evaluation metrics
        self.mof = ClusteringMetrics(metric='mof')
        self.f1 = ClusteringMetrics(metric='f1')
        self.miou = ClusteringMetrics(metric='miou')
        self.save_hyperparameters()
        self.test_cache = []

    def training_step(self, batch, batch_idx):
        features_raw, mask, gt, fname, n_subactions = batch
        with torch.no_grad():
            self.clusters.data = F.normalize(self.clusters.data, dim=-1)
        D = self.layer_sizes[-1]
        B, T, _ = features_raw.shape
        features = F.normalize(self.mlp(features_raw.reshape(-1, features_raw.shape[-1])).reshape(B, T, D), dim=-1)
        codes = torch.exp(features @ self.clusters.T[None, ...] / self.temp)
        codes = codes / codes.sum(dim=-1, keepdim=True)
        with torch.no_grad():  # pseudo-labels from OT
            temp_prior = asot.temporal_prior(T, self.n_clusters, self.rho, features.device)
            opt_codes = asot.segment_asot(features.detach(), self.clusters, mask, eps=self.train_eps, alpha=self.alpha_train, radius=self.radius_gw,
                                          proj_type=self.ub_proj_type, ub_weight=self.ub_train, n_iters=self.n_ot_train, temp_prior=temp_prior)

        loss_ce = -((opt_codes * torch.log(codes + num_eps)) * mask[..., None]).sum(dim=2).mean()
        self.log('train_loss', loss_ce)
        return loss_ce

    def validation_step(self, batch, batch_idx):  # subsample videos
        features_raw, mask, gt, fname, n_subactions = batch
        D = self.layer_sizes[-1]
        B, T, _ = features_raw.shape
        # import pdb; pdb.set_trace()
        features = F.normalize(self.mlp(features_raw.reshape(-1, features_raw.shape[-1])).reshape(B, T, D), dim=-1)

        # log clustering metrics over full epoch
        temp_prior = asot.temporal_prior(T, self.n_clusters, self.rho, features.device)
        segmentation = asot.segment_asot(features, self.clusters, mask, eps=self.eval_eps, alpha=self.alpha_eval, radius=self.radius_gw,
                                         proj_type=self.ub_proj_type, ub_weight=self.ub_eval, n_iters=self.n_ot_eval, temp_prior=temp_prior)
        segments = segmentation.argmax(dim=2)
        self.mof.update(segments, gt, mask)
        self.f1.update(segments, gt, mask)
        self.miou.update(segments, gt, mask)

        # log clustering metrics per video
        metrics = indep_eval_metrics(segments, gt, mask, ['mof', 'f1', 'miou'], exclude_cls=self.exclude_cls)
        self.log('val_mof_per', metrics['mof'])
        self.log('val_f1_per', metrics['f1'])
        self.log('val_miou_per', metrics['miou'])

        # log validation loss
        codes = torch.exp(features @ self.clusters.T / self.temp)
        codes /= codes.sum(dim=-1, keepdim=True)
        pseudo_labels = asot.segment_asot(features, self.clusters, mask, eps=self.train_eps, alpha=self.alpha_train, radius=self.radius_gw,
                                          proj_type=self.ub_proj_type, ub_weight=self.ub_train, n_iters=self.n_ot_train, temp_prior=temp_prior)
        loss_ce = -((pseudo_labels * torch.log(codes + num_eps)) * mask[..., None]).sum(dim=[1, 2]).mean()
        self.log('val_loss', loss_ce)

        # plot qualitative examples of pseduo-labelling and embeddings for 5 videos evenly spaced in dataset
        spacing =  int(self.trainer.num_val_batches[0] / 5)
        if batch_idx % spacing == 0 and wandb.run is not None and self.visualize:
            plot_idx = int(batch_idx / spacing)
            gt_cpu = gt[0].cpu().numpy()

            fdists = squareform(pdist(features[0].cpu().numpy(), 'cosine'))
            fig = plot_matrix(fdists, gt=gt_cpu, colorbar=False, title=fname[0], figsize=(5, 5), xlabel='Frame index', ylabel='Frame index')
            wandb.log({f"val_pairwise_{plot_idx}": fig, "trainer/global_step": self.trainer.global_step})
            fig = plot_matrix(codes[0].cpu().numpy().T, gt=gt_cpu, colorbar=False, title=fname[0], figsize=(10, 5), xlabel='Frame index', ylabel='Action index')
            wandb.log({f"val_P_{plot_idx}": fig, "trainer/global_step": self.trainer.global_step})
            fig = plot_matrix(pseudo_labels[0].cpu().numpy().T, gt=gt_cpu, colorbar=False, title=fname[0], figsize=(10, 5), xlabel='Frame index', ylabel='Action index')
            wandb.log({f"val_OT_PL_{plot_idx}": fig, "trainer/global_step": self.trainer.global_step})
            fig = plot_matrix(segmentation[0].cpu().numpy().T, gt=gt_cpu, colorbar=False, title=fname[0], figsize=(10, 5), xlabel='Frame index', ylabel='Action index')
            wandb.log({f"val_OT_pred_{plot_idx}": fig, "trainer/global_step": self.trainer.global_step})
            plt.close()

            cost_mat = 1. - features @ self.clusters.T
            bal_codes = asot.segment_asot(features, self.clusters, mask, eps=self.eval_eps, alpha=self.alpha_eval, radius=self.radius_gw,
                                            proj_type='const', ub_weight=self.ub_eval, n_iters=self.n_ot_eval, temp_prior=temp_prior)
            nogw_codes = asot.segment_asot(features, self.clusters, mask, eps=self.eval_eps, alpha=0., radius=self.radius_gw,
                                            proj_type=self.ub_proj_type, ub_weight=self.ub_eval, n_iters=self.n_ot_eval, temp_prior=temp_prior)
            fig = plot_segmentation(segments[0], mask[0], name=f'{fname[0]}')
            wandb.log({f"val_segment_{int(batch_idx / spacing)}": wandb.Image(fig), "trainer/global_step": self.trainer.global_step})

            fig = plot_matrix(segmentation[0].cpu().numpy().T, gt=None, colorbar=False, title=None, xlabel='Frame index', ylabel='Action index')
            wandb.log({f"pred_{plot_idx}": fig, "trainer/global_step": self.trainer.global_step})
            fig = plot_matrix(cost_mat[0].cpu().numpy().T, gt=None, colorbar=False, title=None, xlabel='Frame index', ylabel='Action index')
            wandb.log({f"cost_mat_{plot_idx}": fig, "trainer/global_step": self.trainer.global_step})
            fig = plot_matrix(1. - cost_mat[0].cpu().numpy().T, gt=None, colorbar=False, title=None, xlabel='Frame index', ylabel='Action index')
            wandb.log({f"aff_mat_{plot_idx}": fig, "trainer/global_step": self.trainer.global_step})
            fig = plot_matrix(bal_codes[0].cpu().numpy().T, gt=None, colorbar=False, title=None, xlabel='Frame index', ylabel='Action index')
            wandb.log({f"bal_pred_{plot_idx}": fig, "trainer/global_step": self.trainer.global_step})
            fig = plot_matrix(nogw_codes[0].cpu().numpy().T, gt=None, colorbar=False, title=None, xlabel='Frame index', ylabel='Action index')
            wandb.log({f"nogw_{plot_idx}": fig, "trainer/global_step": self.trainer.global_step})
            plt.close()
        return None
    
    def test_step(self, batch, batch_idx):  # subsample videos
        features_raw, mask, gt, fname, n_subactions = batch
        D = self.layer_sizes[-1]
        B, T, _ = features_raw.shape
        features = F.normalize(self.mlp(features_raw.reshape(-1, features_raw.shape[-1])).reshape(B, T, D), dim=-1)

        # log clustering metrics over full epoch
        temp_prior = asot.temporal_prior(T, self.n_clusters, self.rho, features.device)
        indep_codes = asot.segment_asot(features, self.clusters, mask, eps=self.eval_eps, alpha=self.alpha_eval, radius=self.radius_gw,
                                        proj_type=self.ub_proj_type, ub_weight=self.ub_eval, n_iters=self.n_ot_eval, temp_prior=temp_prior)
        segments = indep_codes.argmax(dim=2)
        self.mof.update(segments, gt, mask)
        self.f1.update(segments, gt, mask)
        self.miou.update(segments, gt, mask)

        # log clustering metrics per video
        metrics = indep_eval_metrics(segments, gt, mask, ['mof', 'f1', 'miou'], exclude_cls=self.exclude_cls)
        self.log('test_mof_per', metrics['mof'])
        self.log('test_f1_per', metrics['f1'])
        self.log('test_miou_per', metrics['miou'])

        # cache videos for plotting
        self.test_cache.append([metrics['mof'], segments, gt, mask, fname])

        return None
    
    def on_validation_epoch_end(self):
        mof, pred_to_gt = self.mof.compute(exclude_cls=self.exclude_cls)
        f1, _ = self.f1.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        miou, _ = self.miou.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        self.log('val_mof_full', mof)
        self.log('val_f1_full', f1)
        self.log('val_miou_full', miou)
        self.mof.reset()
        self.f1.reset()
        self.miou.reset()

    def on_test_epoch_end(self):
        mof, pred_to_gt = self.mof.compute(exclude_cls=self.exclude_cls)
        f1, _ = self.f1.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        miou, _  = self.miou.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        self.log('test_mof_full', mof)
        self.log('test_f1_full', f1)
        self.log('test_miou_full', miou)
        if wandb.run is not None and self.visualize:
            for i, (mof, pred, gt, mask, fname) in enumerate(self.test_cache):
                self.test_cache[i][0] = indep_eval_metrics(pred, gt, mask, ['mof'], exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)['mof']
            self.test_cache = sorted(self.test_cache, key=lambda x: x[0], reverse=True)

            for i, (mof, pred, gt, mask, fname) in enumerate(self.test_cache[:10]):
                fig = plot_segmentation_gt(gt, pred, mask, exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt,
                                           gt_uniq=np.unique(self.mof.gt_labels), name=f'{fname[0]}')
                wandb.log({f"test_segment_{i}": wandb.Image(fig), "trainer/global_step": self.trainer.global_step})
                plt.close()
        self.test_cache = []
        self.mof.reset()
        self.f1.reset()
        self.miou.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def fit_clusters(self, dataloader, K):
        with torch.no_grad():
            features_full = []
            self.mlp.eval()
            for features_raw, _, _, _, _ in dataloader:
                B, T, _ = features_raw.shape
                D = self.layer_sizes[-1]
                features = F.normalize(self.mlp(features_raw.reshape(-1, features_raw.shape[-1])).reshape(B, T, D), dim=-1)
                features_full.append(features)
            features_full = torch.cat(features_full, dim=0).reshape(-1, features.shape[2]).cpu().numpy()
            kmeans = KMeans(n_clusters=K).fit(features_full)
            self.mlp.train()
        self.clusters.data = torch.from_numpy(kmeans.cluster_centers_).to(self.clusters.device)
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train representation learning pipeline")

    # FUGW OT segmentation parameters
    parser.add_argument('--alpha-train', '-at', type=float, default=0.3, help='weighting of KOT term on frame features in OT')
    parser.add_argument('--alpha-eval', '-ae', type=float, default=0.3, help='weighting of KOT term on frame features in OT')
    parser.add_argument('--ub-train', '-ut', type=float, default=0.05, help='penalty on balanced classes assumption for training')
    parser.add_argument('--ub-eval', '-ue', type=float, default=0.01, help='penalty on balanced classes assumption for eval')
    parser.add_argument('--proj-type', '-pt', type=str, default='kl', choices=['kl', 'tv', 'const'],
                        help='penalty type for unbalanced problem. kl is default and const is balanced assignment')
    parser.add_argument('--eps-train', '-et', type=float, default=0.06, help='entropy regularization for OT during training')
    parser.add_argument('--eps-eval', '-ee', type=float, default=0.01, help='entropy regularization for OT during val/test')
    parser.add_argument('--radius-gw', '-r', type=float, default=0.04, help='Radius parameter for GW structure loss')
    parser.add_argument('--n-ot-train', '-nt', type=int, nargs='+', default=[25, 15], help='number of outer and inner Sinkhorn iterations for OT (train)')
    parser.add_argument('--n-ot-eval', '-no', type=int, nargs='+', default=[25, 15], help='number of outer and inner Sinkhorn iterations for OT (train)')
    
    # dataset params
    parser.add_argument('--base-path', '-p', type=str, default='/home/users/u6567085/data', help='base directory for dataset')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='dataset to use for training/eval (Breakfast, YTI, FSeval, FS, desktop_assembly)')
    parser.add_argument('--activity', '-ac', type=str, nargs='+', required=True, help='activity classes to select for dataset')
    parser.add_argument('--exclude', '-x', type=int, default=None, help='classes to exclude from evaluation. use -1 for YTI')
    parser.add_argument('--n-frames', '-f', type=int, default=256, help='number of frames sampled per video for train/val')
    parser.add_argument('--std-feats', '-s', action='store_true', help='standardize features per video during preprocessing')
    
    # representation learning params
    parser.add_argument('--n-epochs', '-ne', type=int, default=15, help='number of epochs for training')
    parser.add_argument('--batch-size', '-bs', type=int, default=2, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight-decay', '-wd', type=float, default=1e-4, help='weight decay for optimizer')
    parser.add_argument('--k-means', '-km', action='store_false', help='do not initialize clusters with kmeans default = True')
    parser.add_argument('--layers', '-ls', default=[64, 128, 40], nargs='+', type=int, help='layer sizes for MLP (in, hidden, ..., out)')
    parser.add_argument('--rho', type=float, default=0.1, help='Factor for global structure weighting term')
    parser.add_argument('--n-clusters', '-c', type=int, default=8, help='number of actions/clusters')

    # system/logging params
    parser.add_argument('--val-freq', '-vf', type=int, default=5, help='validation epoch frequency (epochs)')
    parser.add_argument('--gpu', '-g', type=int, default=1, help='gpu id to use')
    parser.add_argument('--wandb', '-w', action='store_true', help='use wandb for logging')
    parser.add_argument('--visualize', '-v', action='store_true', help='generate visualizations during logging')
    parser.add_argument('--seed', type=int, default=0, help='Random seed initialization')
    parser.add_argument('--ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--eval', action='store_true', help='run evaluation on test set only')
    parser.add_argument('--group', type=str, default='base', help='wandb experiment group name')
    args = parser.parse_args()

    pl.seed_everything(args.seed)
        
    data_val = VideoDataset('/home/users/u6567085/data', args.dataset, args.n_frames, standardise=args.std_feats, random=False, action_class=args.activity)
    data_train = VideoDataset('/home/users/u6567085/data', args.dataset, args.n_frames, standardise=args.std_feats, random=True, action_class=args.activity)
    data_test = VideoDataset('/home/users/u6567085/data', args.dataset, None, standardise=args.std_feats, random=False, action_class=args.activity)
    val_loader = DataLoader(data_val, batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=1, shuffle=False)

    if args.ckpt is not None:
        ssl = VideoSSL.load_from_checkpoint(args.ckpt)
    else:
        ssl = VideoSSL(layer_sizes=args.layers, n_clusters=args.n_clusters, alpha_train=args.alpha_train, alpha_eval=args.alpha_eval, ub_train=args.ub_train, ub_eval=args.ub_eval,
                       ub_proj_type=args.proj_type, train_eps=args.eps_train, eval_eps=args.eps_eval, radius_gw=args.radius_gw, n_ot_train=args.n_ot_train, n_ot_eval=args.n_ot_eval,
                       n_frames=args.n_frames, lr=args.learning_rate, weight_decay=args.weight_decay, rho=args.rho, exclude_cls=args.exclude, visualize=args.visualize)

    activity_name = '_'.join(args.activity)
    name = f'{args.dataset}_{activity_name}_{args.group}_seed_{args.seed}'
    logger = pl.loggers.WandbLogger(name=name, project='video_ssl', save_dir='wandb') if args.wandb else None
    trainer = pl.Trainer(devices=[args.gpu], check_val_every_n_epoch=args.val_freq, max_epochs=args.n_epochs, log_every_n_steps=50, logger=logger)

    if args.k_means and args.ckpt is None:
        ssl.fit_clusters(train_loader, args.n_clusters)

    if not args.eval:
        trainer.validate(ssl, val_loader)
        trainer.fit(ssl, train_loader, val_loader)
    trainer.test(ssl, dataloaders=test_loader)
