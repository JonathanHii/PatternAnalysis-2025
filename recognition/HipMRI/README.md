# Improved UNet3D for Prostate MRI Segmentation with Attention, Residual, and ASPP Modules
Author: Jonathan Hii s4883828

This repository presents the development, data augmentation strategies, and training pipeline for an Improved UNet3D architecture, designed for 3D medical image segmentation on the Prostate 3D dataset. The model builds upon the foundational U-Net architecture which has become one of the most influential convolutional network designs in biomedical image segmentation. The improved version implemented here integrates advanced components such as residual connections, squeeze-and-excitation (SE) blocks, attention gates, and atrous spatial pyramid pooling (ASPP) to enhance feature representation, gradient flow, and multi-scale context capture in volumetric data.

For detailed insights into the original 3D U-Net design and medical imaging applications, please refer to the paper:
[Few-Shot Learning for Medical Image Segmentation Using 3D U-Net and Model-Agnostic Meta-Learning (MAML)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11202447/#notes1)

## Table of Contents
- [Data Set](#data-set)
- [Project Goal](#project-goal)
- [File Structure](#file-structure)
- [Model Architecture](#model-architecture)
  - [Base UNet3D Overview](#base-unet3d-overview)
  - [Residual Connections](#residual-connections)
  - [Squeeze-and-Excitation (SE) Blocks](#squeeze-and-excitation-se-blocks)
  - [Attention Gates](#attention-gates)
  - [Atrous Spatial Pyramid Pooling (ASPP)](#atrous-spatial-pyramid-pooling-aspp)
- [Training](#training)
- [Training Results](#training-results)
- [Dependencies](#dependencies)
- [References](#references)

## Data Set
The project uses the downsampled Prostate 3D dataset consisting of paired MRI volumes and segmentation masks stored in NIfTI format (`.nii.gz`).  
Each image–label pair is identified using matching filename keys (`*_LFOV` for MRI, `*_SEMANTIC` for label). 

Data pairs are automatically matched using the `find_pairs()` utility, which scans the image and label directories and returns valid `(key, image_path, label_path)` tuples.

## Project Goal
The goal of this project is to segment the downsampled Prostate 3D MRI dataset using an **Improved UNet3D** model that integrates **residual connections**, **squeeze-and-excitation (SE) blocks**, **attention gates**, and **ASPP modules**.  
The model aims to achieve a minimum Dice Similarity Coefficient (DSC) of **0.7** for all labels on the test set, ensuring accurate and reliable prostate boundary segmentation.

## File Structure
This repository consists of the following four major files:
- `dataset.py` - Loads and preprocesses images, returns tensors and labels for training and validation.
- `modules.py` - Defines model architectures, losses, and metric functions used during training and evaluation.
- `train.py` - Training and validation loop, handles optimizer/scheduler, checkpointing, and logging.
- `predict.py` - Loads a trained checkpoint and runs inference on new inputs

## Model Architecture
### Base UNet3D Overview

### Residual Connections

### Squeeze-and-Excitation (SE) Blocks

### Attention Gates

### Atrous Spatial Pyramid Pooling (ASPP)
## Training


## Training Results

## Dependencies
- Python 3.7+
- PyTorch 1.10+
- CUDA (for GPU support, optional)
- torchvision
- numpy
- tqdm
- matplotlib
- nibabel

## References
- O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2016, ser. Lecture Notes in Computer Science, S. Ourselin, L. Joskowicz, M. R. Sabuncu,
G. Unal, and W. Wells, Eds. Cham: Springer International Publishing, 2016, pp. 424–432.

