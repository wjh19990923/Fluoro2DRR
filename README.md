# Fluoro2DRR

Fluoroscopic-DRR image translation modules for
Automated 2D/3D Fluoroscopic Registration for Knee Joint Kinematics

This repository contains the implementation of **Fluoro2DRR**, a image translation pipeline for denoising and demosaicking fluoroscopic images (X-ray).
The method is designed to robustly register low-dose, noise-contaminated fluoroscopy to 3D bone models by reformulating the registration problem into a near mono-modal DRR-to-DRR optimization.

This work accompanies the ISBI 2026 submission:

> **Automated 2D/3D Fluoroscopic Registration for Knee Joint Kinematics Using Fluoroscopic-to-DRR Translation**  
> J. Wang, R. Surbeck, S. Ćuković, W. R. Taylor, X. Li

To understand what we are doing, please visit our group page: https://movement.ethz.ch/

---

## Overview

Conventional fluoroscopic 2D/3D registration suffers from severe modality discrepancies between real X-ray images and digitally reconstructed radiographs (DRRs), especially under low-dose acquisition and marker/sensor artefacts.  
**Fluoro2DRR** addresses this challenge by:

1. Translating fluoroscopic images into DRR-like representations using a two-stage deep learning framework.
2. Converting a multi-modal, non-convex registration problem into a smoother mono-modal DRR-to-DRR optimization.
3. Enabling fully automated and robust pose estimation with sub-millimetre accuracy.

The pipeline combines learning-based image translation with model-based differentiable registration.

---
## Method
- **Fluoro2DRR Image Translation**
   - UNet-based GAN for denoising and artefact suppression
   - Super-resolution (DRUNet) for structural detail restoration and demosaicking
---

## Dataset

The method was validated on a large single-plane fluoroscopic dataset of knee joint motion:

- **Subjects**: 28 healthy participants  
- **Images**: 29,950 calibrated fluoroscopic frames  
- **Ground Truth**: Expert-validated 3D poses with sub-millimetre accuracy  
- **Activities**: Squatting, level walking, and daily functional movements  

All data splits were performed **subject-wise** to avoid data leakage.

> ⚠️ Due to ethical and privacy constraints, this fluoroscopic dataset has not be publicly released yet.

---

## Results

Key performance highlights:

- **Translation accuracy**: 0.23 mm (in-plane)
- **Rotation accuracy**: 0.67° (geodesic error)
- **Success rate**: >97% sub-millimetre convergence
- Significant improvement over registration using raw fluoroscopic images
- Increased capture range and robustness under severe noise and artefacts

---

## 🧪 Ablation Studies

The repository includes ablation experiments demonstrating:

- The contribution of each loss term (L1, SSIM, LPIPS, segmentation)
- The impact of super-resolution demosaicking on registration accuracy
- Comparison of different super-resolution backbones (SR-ResNet, DRUNet, NAFNet)

---

## ⚙️ Installation

```bash
git clone https://github.com/wjh19990923/Fluoro2DRR.git
cd Fluoro2DRR
pip install -r requirements.txt
