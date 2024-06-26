# Video Reconstruction Through Dynamic Scattering Medium via Spatio-Temporal Diffusion Models

## Supplementary Videos
<p align="center" width="100%">
    <img src='./supplementary videos/ReconstructionResultsforUCF101.gif' width='49%'>
    <img src='./supplementary videos/ReconstructionResultsforVISEM-Tracking.gif' width='49%'>
</p>

## Overview

<p align="center" width="100%">
    <img width="67%" src="./figures/cover.jpg">
</p>

<p align="center" width="100%">
    <img width="10%" src="./figures/UCF_measurement.gif">
    <img width="10%" src="./figures/UCF_output.gif">
    <img width="20%" src="./figures/VISEM_measurement.gif">
    <img width="20%" src="./figures/VISEM_output.gif">
</p>

## Abstract
In many optical imaging applications, optical scattering randomly disturbs light propagation, posing the challenge of recovering well-resolved images from scattered signals. Existing computational approaches for inverse scattering solvers rely on statistical or empirical constraints in the spatial domain, often failing to recover temporally coherent frames of dynamic scenes. Here, with a closed-form forward model for a dynamic scattering medium of finite thickness, we implement a video diffusion posterior sampling scheme with temporal attention, maximizing the statistical correlation between frames and scattered signals. By incorporating temporal correlation, our method accurately captures high-resolution components that are missed in spatial-domain operations alone. Furthermore, with explicit modeling of spatio-temporal correlation, our approach demonstrates adaptability across diverse scenarios, including various scene types, scattering media thicknesses, and scene-medium distances. Numerical validation and real experimental results using various datasets and optical configurations validate the effectiveness of our approach. To the best of our knowledge, this is the first video diffusion model to exploit correlations in both spatial and temporal domains for solving the inverse scattering problem.


## Prerequisites
- python 3.10

- pytorch 1.13.1

- CUDA 11.7

It is okay to use lower version of CUDA with proper pytorch version.

Ex) CUDA 10.2 with pytorch 1.7.0

<br />

## Getting started 

### 1) Clone the repository

```
git clone https://github.com/star-kwon/VDPS

cd VDPS
```

<br />

### 2) Download pretrained checkpoint and sample videos
From the [link](https://drive.google.com/drive/folders/1-Zu7GL2dooGFJYEO34s9U0J03LKqd6I6?usp=sharing), download the checkpoints and paste it to ./models/, download the samples and paste it to ./scatter samples/
```
mkdir models
mkdir scatter samples
mv {MODEL_DOWNLOAD_DIR}/{CHECKPOINT NAME} ./{models}/
mv {SAMPLE_DOWNLOAD_DIR}/{SAMPLES} ./{scatter samples}/
```
{DOWNLOAD_DIR} is the directory that you downloaded checkpoint to.
{PASTE_DIR} is the directory that you should paste to.

<br />

### 3) Set environment
Install dependencies

```
conda create -n VDPS python=3.10

conda activate VDPS

pip install -r requirements.txt

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

<br />

### 4) Test
Run test code for simulation experiments.
The real measurement reconstruction results in video format are in the 'video_results' folder.

```
python test_UCF.py
python test_VISEM.py
```

<br />
