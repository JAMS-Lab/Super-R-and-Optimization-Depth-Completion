# 🧩 Depth Completion With Super-Resolution and Cross-Modality Optimization

This repository provides the implementation of our depth completion framework, designed to generate high-quality dense depth maps from sparse or incomplete input.

## 🔍 Overview

Our method integrates single-image relative depth estimation, super-resolution refinement, and global optimization into a unified pipeline. It is tailored for robust depth completion under varied conditions and across datasets.

### ✨ Key Features

- **Relative Depth Estimation**: Predicts coarse depth from a single RGB image.
- **Super-Resolution Refinement**: Utilizes Fast Fourier Convolution (FFC) and Gradient-weighted Symmetric Feature Transmission (GSFT) to enhance depth detail with RGB guidance.
- **Cross-Modality Optimization**: Fuses refined depth with sparse ground truth through a global optimization step.
- **Strong Generalization**: Avoids overfitting to specific sensor noise or corruption patterns.

The approach is validated on NYU-Depth V2 and SUN RGB-D benchmarks, showing superior accuracy and resolution compared to prior methods. 

---

## 📄 Citation

If you find this work helpful, please cite:

```bibtex
@article{zhong2025depth,
  title={Depth Completion With Super-Resolution and Cross-Modality Optimization},
  author={Zhong, Ju and Jiang, Aimin and Liu, Chang and Xu, Ning and Zhu, Yanping},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}
