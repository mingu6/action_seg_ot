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
import seg_ot
from metrics import ClusteringMetrics, indep_eval_metrics

num_eps = 1e-11


def contrastive(features, pos_radius, mask):
    B, N, _ = features.shape
    frame_id = np.arange(N)[None, :].repeat(B, 0)
    pos_ind = np.random.randint(np.maximum(frame_id - pos_radius, 0.), np.minimum(frame_id + pos_radius, N))
    pos_ind = torch.from_numpy(pos_ind).to(features.device)
    loss = 0.
    for b in range(B):
        contrastive = torch.exp(features[b] @ features[b][pos_ind[b]].T)
        loss -= torch.log(torch.diag(contrastive) / contrastive.sum(dim=1)).mean()
    return loss / B


class VideoSSL(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=1e-4, layer_sizes=[64, 128, 40], n_clusters=20, alpha=0.2, n_ot_train=[5, 3], n_ot_eval=[25, 10],
                 train_eps=0.1, eval_eps=0.02, inter_vid_cost=0.3, ub_proj_type='kl', ub_weight=0.1, temp=0.1, ema=0., gw_radius=5, learn_clusters=False,
                 n_frames=256, rho=0.1, prior_ep=10, prior_fac=0.25, entropy_train=0.25, bn=True):
        super().__init__()
        self.lr = lr
        self.n_clusters = n_clusters
        self.weight_decay = weight_decay
        self.layer_sizes = layer_sizes
        self.alpha = alpha
        self.n_ot_train = n_ot_train
        self.n_ot_eval = n_ot_eval
        self.temp = temp
        self.train_eps = train_eps  # entropy weight for OT
        self.eval_eps = eval_eps  # entropy weight for OT
        self.gw_radius = gw_radius
        self.ub_proj_type = ub_proj_type
        self.inter_vid_cost = inter_vid_cost
        self.ub_weight = ub_weight
        self.learn_clusters = learn_clusters
        self.n_frames = n_frames
        self.rho = rho
        self.prior_ep = prior_ep
        self.prior_fac = prior_fac
        self.entropy_train = entropy_train
        self.bn = bn
        # initialize MLP
        if bn:
            layers = [nn.Sequential(nn.Linear(sz, sz1), nn.ReLU(), nn.BatchNorm1d(sz1)) for sz, sz1 in zip(layer_sizes[:-2], layer_sizes[1:-1])]
        else:
            layers = [nn.Sequential(nn.Linear(sz, sz1), nn.ReLU()) for sz, sz1 in zip(layer_sizes[:-2], layer_sizes[1:-1])]
        layers += [nn.Linear(layer_sizes[-2], layer_sizes[-1])]
        self.mlp = nn.Sequential(*layers)
        # initialize cluster centers/codebook
        d = self.layer_sizes[-1]
        self.clusters = nn.parameter.Parameter(data=F.normalize(torch.randn(self.n_clusters, d), dim=-1), requires_grad=learn_clusters)
        # initialize evaluation metrics
        self.nmi = ClusteringMetrics(metric='nmi')
        self.ari = ClusteringMetrics(metric='ari')
        self.mof = ClusteringMetrics(metric='mof')
        self.f1 = ClusteringMetrics(metric='f1')
        self.f1_w = ClusteringMetrics(metric='f1_w')
        self.iou = ClusteringMetrics(metric='iou')
        # EMA stuff
        self.mlp_ma = deepcopy(self.mlp)
        self.ema = EMA(ema)
        self.save_hyperparameters()

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
            features_opt = F.normalize(self.mlp_ma(features_raw.reshape(-1, features_raw.shape[-1])).reshape(B, T, D), dim=-1)
            opt_codes = seg_ot.segment_indep(features_opt, self.clusters, mask, eps=self.train_eps, alpha=self.alpha, radius=self.gw_radius,
                                             proj=self.ub_proj_type, proj_weight=self.ub_weight, n_iters=self.n_ot_train, rho=self.rho)

        loss_ce = -((opt_codes * torch.log(codes + num_eps)) * mask[..., None]).sum(dim=2).mean()
        class_rep = (codes * mask[..., None]).mean(dim=[0,1])
        entropy_reg = (class_rep * torch.log(class_rep + num_eps)).sum()
        loss = loss_ce + self.entropy_train * entropy_reg
        self.log('train_ce_loss', loss_ce)
        self.log('train_entropy', entropy_reg)
        self.log('train_loss', loss)
        update_moving_average(self.ema, self.mlp_ma, self.mlp)
        return loss

    def validation_step(self, batch, batch_idx):  # subsample videos
        features_raw, mask, gt, fname, n_subactions = batch
        D = self.layer_sizes[-1]
        B, T, _ = features_raw.shape
        features = F.normalize(self.mlp(features_raw.reshape(-1, features_raw.shape[-1])).reshape(B, T, D), dim=-1)

        # log clustering metrics over full epoch
        indep_codes = seg_ot.segment_indep(features, self.clusters, mask, eps=self.eval_eps, alpha=self.alpha, radius=self.gw_radius,
                                           proj=self.ub_proj_type, proj_weight=self.ub_weight, n_iters=self.n_ot_eval, rho=self.rho)
        segments = indep_codes.argmax(dim=2)
        self.nmi.update(segments, gt, mask)
        self.ari.update(segments, gt, mask)
        self.mof.update(segments, gt, mask)
        self.f1.update(segments, gt, mask)
        self.f1_w.update(segments, gt, mask)
        self.iou.update(segments, gt, mask)

        # log clustering metrics per video
        nmi_per = indep_eval_metrics(segments, gt, mask, 'nmi')
        ari_per = indep_eval_metrics(segments, gt, mask, 'ari')
        mof_per = indep_eval_metrics(segments, gt, mask, 'mof')
        f1_per = indep_eval_metrics(segments, gt, mask, 'f1')
        f1_w_per = indep_eval_metrics(segments, gt, mask, 'f1_w')
        iou_per = indep_eval_metrics(segments, gt, mask, 'iou')
        self.log('val_nmi_per', nmi_per)
        self.log('val_ari_per', ari_per)
        self.log('val_mof_per', mof_per)
        self.log('val_f1_per', f1_per)
        self.log('val_f1_weird_per', f1_w_per)
        self.log('val_iou_per', iou_per)

        # log validation loss

        codes = torch.exp(features @ self.clusters.T / self.temp)
        codes /= codes.sum(dim=-1, keepdim=True)
        features_ma = F.normalize(self.mlp_ma(features_raw.reshape(-1, features_raw.shape[-1])).reshape(B, T, D), dim=-1)
        pseudo_lab = seg_ot.segment_indep(features_ma, self.clusters, mask, eps=self.train_eps, alpha=self.alpha, radius=self.gw_radius,
                                          proj=self.ub_proj_type, proj_weight=self.ub_weight, n_iters=self.n_ot_train, rho=self.rho)
        loss_ce = -((pseudo_lab * torch.log(codes + num_eps)) * mask[..., None]).sum(dim=[1, 2]).mean()
        self.log('val_ce_loss', loss_ce)

        gt_change = np.where((np.diff(gt[0].cpu().numpy()) != 0))[0] + 1
        # plot qualitative examples of pseduo-labelling and embeddings for 5 videos evenly spaced
        spacing =  int(self.trainer.num_val_batches[0] / 5)
        if batch_idx % spacing == 0 and wandb.run is not None:
            img_idx = int(batch_idx / spacing)
            # pairwise intra-video cosine distances
            fdists = squareform(pdist(features[0].cpu().numpy(), 'cosine'))
            fdists = np.nan_to_num(fdists)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            plot1 = ax.matshow(fdists)
            for ch in gt_change:
                ax.axvline(ch, color='red')
            plt.colorbar(plot1, ax=ax)
            ax.set_title(fname[0])
            ax.set_xlabel('Frame idx')
            ax.set_ylabel('Frame idx')
            wandb.log({f"val_pairwise_{img_idx}": fig, "trainer/global_step": self.trainer.global_step})
            plt.close()

            # P matrix of normalized feature/cluster similarities

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            plot1 = ax.matshow(codes[0].cpu().numpy().T)
            for ch in gt_change:
                ax.axvline(ch, color='red')
            ax.set_aspect('auto')
            plt.colorbar(plot1, ax=ax)
            ax.set_title(fname[0])
            ax.set_xlabel('Frame idx')
            ax.set_ylabel('Cluster idx')
            wandb.log({f"val_P_{img_idx}": fig, "trainer/global_step": self.trainer.global_step}) 
            plt.close()

            # Q matrix of codes from OT pseudo-labelling (train/soft)

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            plot1 = ax.matshow(pseudo_lab[0].cpu().numpy().T)
            for ch in gt_change:
                ax.axvline(ch, color='red')
            ax.set_aspect('auto')
            plt.colorbar(plot1, ax=ax)
            ax.set_title(fname[0])
            ax.set_xlabel('Frame idx')
            ax.set_ylabel('Cluster idx')
            wandb.log({f"val_OT_PL_{img_idx}": fig, "trainer/global_step": self.trainer.global_step}) 
            plt.close()

            # Q matrix of codes from OT pseudo-labelling (val/test, harder)

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            plot1 = ax.matshow(indep_codes[0].cpu().numpy().T)
            for ch in gt_change:
                ax.axvline(ch, color='red')
            ax.set_aspect('auto')
            plt.colorbar(plot1, ax=ax)
            ax.set_title(fname[0])
            ax.set_xlabel('Frame idx')
            ax.set_ylabel('Cluster idx')
            wandb.log({f"val_OT_pred_{img_idx}": fig, "trainer/global_step": self.trainer.global_step}) 
            plt.close()
        return None
    
    def test_step(self, batch, batch_idx):  # subsample videos
        features_raw, mask, gt, fname, n_subactions = batch
        D = self.layer_sizes[-1]
        B, T, _ = features_raw.shape
        features = F.normalize(self.mlp(features_raw.reshape(-1, features_raw.shape[-1])).reshape(B, T, D), dim=-1)

        # log clustering metrics over full epoch
        r_test = self.gw_radius / self.n_frames * T
        indep_codes = seg_ot.segment_indep(features, self.clusters, mask, eps=self.eval_eps, alpha=self.alpha, radius=r_test,
                                           proj=self.ub_proj_type, proj_weight=self.ub_weight, n_iters=self.n_ot_eval, rho=self.rho)
        segments = indep_codes.argmax(dim=2)
        self.nmi.update(segments, gt, mask)
        self.ari.update(segments, gt, mask)
        self.mof.update(segments, gt, mask)
        self.f1.update(segments, gt, mask)
        self.f1_w.update(segments, gt, mask)
        self.iou.update(segments, gt, mask)

        # log clustering metrics per video
        nmi_per = indep_eval_metrics(segments, gt, mask, 'nmi')
        ari_per = indep_eval_metrics(segments, gt, mask, 'ari')
        mof_per = indep_eval_metrics(segments, gt, mask, 'mof')
        f1_per = indep_eval_metrics(segments, gt, mask, 'f1')
        f1_w_per = indep_eval_metrics(segments, gt, mask, 'f1_w')
        iou_per = indep_eval_metrics(segments, gt, mask, 'iou')
        self.log('test_nmi_per', nmi_per)
        self.log('test_ari_per', ari_per)
        self.log('test_mof_per', mof_per)
        self.log('test_f1_per', f1_per)
        self.log('test_f1_weird_per', f1_w_per)
        self.log('test_iou_per', iou_per)
        return None
    
    def on_validation_epoch_end(self):
        self.log('val_nmi_full', self.nmi.compute())
        self.log('val_ari_full', self.ari.compute())
        mean_mof, tp_count, n_frames = self.mof.compute()
        mean_f1, precision, recall, n_videos, segments_count = self.f1.compute()
        mean_f1_w, precision_w, recall_w, n_videos_w, segments_count_w = self.f1_w.compute()
        mean_iou, tp_count1, union_count = self.iou.compute()
        self.log('val_mof_full', mean_mof)
        self.log('val_f1_full', mean_f1)
        self.log('val_f1_weird_full', mean_f1_w)
        self.log('val_iou_full', mean_iou)
        self.nmi.reset()
        self.ari.reset()
        self.mof.reset()
        self.f1.reset()
        self.f1_w.reset()
        self.iou.reset()

    def on_test_epoch_end(self):
        self.log('test_nmi_full', self.nmi.compute())
        self.log('test_ari_full', self.ari.compute())
        mean_mof, tp_count, n_frames = self.mof.compute()
        mean_f1, precision, recall, n_videos, segments_count = self.f1.compute()
        mean_f1_w, precision_w, recall_w, n_videos_w, segments_count_w = self.f1_w.compute()
        mean_iou, tp_count1, union_count = self.iou.compute()
        self.log('test_mof_full', mean_mof)
        self.log('test_tp_full', tp_count)
        self.log('test_n_frames', n_frames)
        self.log('test_f1_full', mean_f1)
        self.log('test_f1_weird_full', mean_f1_w)
        self.log('test_prec_full', precision)
        self.log('test_rec_full', recall)
        self.log('test_n_videos', n_videos)
        self.log('test_n_segments_gt', segments_count)
        self.log('test_iou_full', mean_iou)
        self.log('test_union_full', union_count)
        self.nmi.reset()
        self.ari.reset()
        self.mof.reset()
        self.f1.reset()
        self.f1_w.reset()
        self.iou.reset()

    def on_train_epoch_end(self):
        if self.current_epoch == self.prior_ep:
            self.rho *= self.prior_fac

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
    

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
    

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train representation learning pipeline")
    parser.add_argument('--alpha', '-a', type=float, default=0.5, help='weighting of KOT term on frame features in OT')
    parser.add_argument('--ub-weight', '-uw', type=float, default=0.03, help='penalty on balanced classes assumption')
    parser.add_argument('--eps-train', '-et', type=float, default=0.06, help='entropy regularization for OT during training')
    parser.add_argument('--eps-eval', '-ee', type=float, default=0.02, help='entropy regularization for OT during val/test')
    parser.add_argument('--inter-vid-cost', '-ic', type=float, default=0.1, help='inter-video cost for coupled OT training')
    parser.add_argument('--radius-gw', '-r', type=int, default=10, help='Radius parameter for GW structure loss')
    parser.add_argument('--n-ot-train', '-nt', type=int, nargs='+', default=[25, 15], help='number of outer and inner Sinkhorn iterations for OT (train)')
    parser.add_argument('--n-ot-eval', '-no', type=int, nargs='+', default=[25, 15], help='number of outer and inner Sinkhorn iterations for OT (train)')
    
    parser.add_argument('--n-frames', '-f', type=int, default=256, help='number of frames sampled per video for train/val')
    parser.add_argument('--std-feats', '-s', action='store_true', help='standardize features per video during preprocessing')
    
    parser.add_argument('--n-epochs', '-ne', type=int, default=15, help='number of epochs for training')
    parser.add_argument('--batch-size', '-bs', type=int, default=2, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight-decay', '-wd', type=float, default=1e-4, help='weight decay for optimizer')
    parser.add_argument('--entropy-train', '-ht', type=float, default=0.25, help='entropy regularization weight for training loss')
    parser.add_argument('--batch-norm', '-bn', action='store_false', help='do not use batch normalization: default = True')
    parser.add_argument('--k-means', '-km', action='store_false', help='do not initialize clusters with kmeans default = True')
    parser.add_argument('--layers', '-ls', default=[64, 128, 40], nargs='+', type=int, help='layer sizes for MLP (in, hidden, ..., out)')

    parser.add_argument('--ema', '-em', type=float, default=0.99, help='EMA weight (0 is no moving average, only most recent)')
    parser.add_argument('--rho', type=float, default=0.1, help='Factor for global structure weighting term')
    parser.add_argument('--prior-epoch', '-ps', type=int, default=10, help='Epoch number to reduce prior structure weight')
    parser.add_argument('--prior-factor', '-pf', type=float, default=0.25, help='Factor used for reducing prior structure term weight')

    parser.add_argument('--n-clusters', '-c', type=int, default=8, help='number of actions/clusters')
    parser.add_argument('--clusters-learn', '-lc', action='store_true', help='allow clusters to be learnable parameters')

    parser.add_argument('--base-path', '-p', type=str, default='/home/users/u6567085/data', help='base directory for dataset')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='dataset to use for training/eval (Breakfast, YTI, FSeval, FS, desktop_assembly)')
    parser.add_argument('--activity', '-ac', type=str, nargs='+', required=True, help='activity classes to select for dataset')
    parser.add_argument('--val-freq', '-vf', type=int, default=5, help='validation epoch frequency (epochs)')
    parser.add_argument('--gpu', '-g', type=int, default=1, help='gpu id to use')
    parser.add_argument('--wandb', '-w', action='store_true', help='use wandb for logging')
    parser.add_argument('--seed', type=int, default=0, help='Random seed initialization')
    parser.add_argument('--ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--eval', action='store_true', help='run evaluation on test set only')
    parser.add_argument('--group', type=str, default='', help='wandb experiment group name')
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    if args.ckpt is not None:
        ssl = VideoSSL.load_from_checkpoint(args.ckpt)
    else:
        ssl = VideoSSL(layer_sizes=args.layers, n_clusters=args.n_clusters, alpha=args.alpha, ub_weight=args.ub_weight, train_eps=args.eps_train, eval_eps=args.eps_eval,
                    gw_radius=args.radius_gw, n_ot_train=args.n_ot_train, n_ot_eval=args.n_ot_eval, inter_vid_cost=args.inter_vid_cost,
                    learn_clusters=args.clusters_learn, n_frames=args.n_frames, lr=args.learning_rate, weight_decay=args.weight_decay, ema=args.ema,
                    prior_ep=args.prior_epoch, prior_fac=args.prior_factor, rho=args.rho, entropy_train=args.entropy_train, bn=args.batch_norm)

    activity_name = '_'.join(args.activity)
    name = f'{args.dataset}_{activity_name}_{args.group}_seed_{args.seed}'
    logger = pl.loggers.WandbLogger(name=name, project='video_ssl', group='main_results', save_dir='wandb') if args.wandb else None
    trainer = pl.Trainer(devices=[args.gpu], check_val_every_n_epoch=args.val_freq, max_epochs=args.n_epochs, log_every_n_steps=50, logger=logger)
    
    data_val = VideoDataset('/home/users/u6567085/data', args.dataset, args.n_frames, standardise=args.std_feats, random=False, action_class=args.activity)
    data_train = VideoDataset('/home/users/u6567085/data', args.dataset, args.n_frames, standardise=args.std_feats, random=True, action_class=args.activity)
    data_test = VideoDataset('/home/users/u6567085/data', args.dataset, None, standardise=args.std_feats, random=False, action_class=args.activity)
    val_loader = DataLoader(data_val, batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=1, shuffle=False)

    if args.k_means and args.ckpt is None:
        ssl.fit_clusters(train_loader, args.n_clusters)

    if not args.eval:
        trainer.validate(ssl, val_loader)
        trainer.fit(ssl, train_loader, val_loader)
    trainer.test(ssl, dataloaders=test_loader)
