# Temporally Consistent Unbalanced Optimal Transport for Unsupervised Action Segmentation

This repo contains a reference implementation of our proposed method Action Segmentation Optimal Transport (ASOT), accepted to CVPR 2024. We also include our full training/evaluation pipelines for the unsupervised learning experiments in the paper.

System figure explaining ASOT

Regular OT pseudo-labels, ASOT pseudo-labels

## 1. Post-processing example

We provide a self-contained example which shows how ASOT is used for post-processing in `examples/`. We have an unsupervised and supervised example from MS-TCN++.

## 2. Unsupervised Learning

Some setup is required to run the unsupervised learning pipeline. 

### 2.1 Datasets

See here. Download pre-extracted features into the following structure

### 2.2 Dependencies

`numpy scipy scikit-learn matplotlib pytorch pytorch-lightning wandb`

### 2.3 Run train/eval pipeline

Examples bf/yti/FS/DA etc
