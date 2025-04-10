# Diffusion-based Video Reconstruction Beyond Dynamic Scattering Layer

This repository is the official implementation of "Diffusion-based Video Reconstruction Beyond Dynamic Scattering Layer", led by

[Taesung Kwon](https://star-kwon.github.io/), [Gookho Song](https://scholar.google.com/citations?user=YJQV1tgAAAAJ&hl=en), [Yoosun Kim](https://scholar.google.com/citations?user=AHILv2QAAAAJ&hl=en), [Jeongsol Kim](https://jeongsol.dev/), [Jong Chul Ye](https://bispl.weebly.com/professor.html), and [Mooseok Jang](https://scholar.google.com/citations?user=QYPGDkAAAAAJ&hl=en).

![main figure](figures/cover.jpg)

<p align="center" width="100%">
    <img width="20%" src="./figures/UCF_measurement.gif" alt="UCF Measurement">
    <img width="20%" src="./figures/UCF_output.gif" alt="UCF Output">
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <img width="20%" src="./figures/VISEM_measurement.gif" alt="VISEM Measurement">
    <img width="20%" src="./figures/VISEM_output.gif" alt="VISEM Output">
</p>

---
## 🔥 Summary
**Imaging through scattering layer** is inherently challenging, as even a thin scattering layer can randomly perturb light propagation, rendering the scene opaque and obscuring objects behind it.
In this work, we propose an approximate forward model tailored for a **dynamic scattering layer with finite thickness**.

To reconstruct videos through such layer, we introduce **VDPS**, a plug-and-play inverse scattering solver powered by video diffusion models.

Our main contributions are as follows:

1. **We demonstrate that leveraging temporal correlations significantly improves spatial reconstruction, achieving state-of-the-art restoration performance.**
2. **We propose a novel inference-time optimization strategy that enables simultaneous estimation of forward-model parameters without requiring additional training.**
3. **Beyond inverse scattering, we show that VDPS is also effective in video dehazing, deblurring, inpainting, and blind PSF restoration via Zernike coefficient estimation.**

## 🛠️ Setup
First, create your environment. We recommend using the following comments. 
```
git clone https://github.com/star-kwon/VDPS.git
cd VDPS
conda env create -f environment.yaml
```

## 📂 Data preperation

For reproducibility, we provide the same dataset used in the official implementation.
Please download the datasets from the following [download link](https://drive.google.com/file/d/1GkuzyTmId2LjGu3bOQNKTm0MDA8aqTSS/view?usp=sharing).

After downloading, place the data folder in the root directory of the repository.
The directory structure should be as follows:
```
VDPS/
├── data/
│   ├── DAVIS_Test/
│   ├── UCF101_Test/
│   ├── Real_Test/
│   └── VISEM_Test/
├── ...
```

## 🎯 Download Checkpoints

We also provide the checkpoints used in the official implementation.
Please download the checkpoints from the following [download link](https://drive.google.com/file/d/1BHm6v41HT7AHxV18qYyK_DM5e1n-KIpi/view?usp=sharing).

After downloading, place the checkpoints in results folder in the root directory of the repository.
The directory structure should be as follows:
```
VDPS/
├── results/
│   ├── model-100.1.pt
│   └── model-100.2.pt
├── ...
```

* `model-100.1.pt` is the pretrained weight of the video diffusion model for the UCF101 and DAVIS datasets.
* `model-100.2.pt` is the pretrained weight of the video diffusion model for the VISEM-Tracking dataset.


## ▶️ Usage

### Sec. 3.2 .Restoration from known degradation
```
python -m eval --deg physics --dist 5.0 --sigma 1.0 --dataset DAVIS
```
Supported datasets: 'DAVIS', 'UCF101', 'VISEM'

### Sec. 3.2. Restoration from blind degradation
```
python -m eval_blind --dist_min 2.5 --dist_max 5 --sigma_min 0.5 --sigma_max 1
```
[Note]
If min and max values are set to the same number, the corresponding parameter is treated as fixed during restoration.
For example:
* dist_min 5.0 --dist_max 5.0 → distance is fixed
* sigma_min 1.0 --sigma_max 1.0 → sigma is fixed

### Sec. 3.3-4. Restoration from real measurements
```
python -m eval_blind_real --dataset Real
```

### Sec. 3.5. Restoration from diverse forward models
```
python -m eval --deg dehaze
```
Supported degradations: 'dehaze', 'inpaint', 'blur'

### Sec. 3.5. Restoration from blind PSF using Zernike coefficents
```
python -m eval_blind_zernike

python -m eval_blind_zernike_varying
```
* `eval_blind_zernike` restores the PSF by simultaneously estimating static Zernike coefficients.
* `eval_blind_zernike_varying` restores the PSF by simultaneously estimating time-varying Zernike coefficients.

## 🎥 Supplementary Videos
<p align="center" width="100%">
    <img src='./supplementary videos/ReconstructionResultsforUCF101.gif' width='49%'>
    <img src='./supplementary videos/ReconstructionResultsforVISEM-Tracking.gif' width='49%'>
</p>