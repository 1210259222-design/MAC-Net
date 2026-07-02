# MAC-Net: Mask-Guided Dual-Channel Neural Network for DAS-VSP Data Reconstruction

This repository contains the official PyTorch implementation and the empirical noise bank for the paper: **Mask-Guided Dual-Channel Neural Network for DAS-VSP Data Reconstruction** (accepted/published in *IEEE Sensors Journal*).

## Repository Structure

    ├── generate_dataset.py       # Script for generating synthetic elastic wave forward modeling data
    ├── train.py                  # Main training script for MAC-Net
    ├── evaluate_synthetic.py     # Evaluation script for synthetic data (Zero-filled & Noise-filled scenarios)
    ├── noise_bank/               # Empirical coupling and background noise library
    │   ├── real_noise_manual.npy
    │   ├── real_noise_purified.h5
    │   ├── simulated_noise_custom.h5
    │   └── simulated_noise_multiregion.h5
    └── README.txt

## Data Availability Statement

The synthetic dataset generation pipeline (`generate_dataset.py`) and the empirical noise library (`noise_bank/`) used to train and validate MAC-Net are fully open-sourced in this repository. Due to strict non-disclosure agreements (NDA) with the exploration project stakeholders, the raw field DAS-VSP data and its specific application scripts are withheld. However, the provided synthetic data engine and noise bank are completely sufficient to independently reproduce the proposed methodology and evaluate the network's wavefield reconstruction capabilities as described in the synthetic experiments of our paper.

## Dependencies

The code is developed and tested under the following environment:
* Python >= 3.8
* PyTorch >= 1.10.0
* NumPy
* h5py
* Matplotlib
* scikit-image
* pytorch-msssim
* numba

Install the required packages via pip:

    pip install torch numpy h5py matplotlib scikit-image pytorch-msssim numba
    
## ⚠️ Dataset Download Instructions
The raw scripts expect a `noise_bank/` directory in the root folder. Please download the empirical noise datasets from the [Releases](你的Release链接地址) page, extract them, and place them inside a newly created `noise_bank/` folder before running the code.
## Quick Start

### 1. Dataset Generation
Construct the synthetic dataset using the elastic wave forward modeling engine and the provided noise bank. 

    python generate_dataset.py

*This will output the required `.h5` format training and validation datasets.*

### 2. Model Training
Train the MAC-Net model from scratch. The script includes the Domain-Adaptive Training Strategy and the Depth-Weighted L1 Loss.

    python train.py

*Trained model weights and convergence logs will be saved in the designated output directory.*

### 3. Evaluation
Evaluate the model's performance on the synthetic blind zones (45-50 traces) coupled with extreme noise. The script calculates Global SNR, Hole SNR, and SSIM.

    python evaluate_synthetic.py

## Citation

If you find this code or the noise bank useful for your research, please cite our paper:

    @article{bai2026macnet,
      title={Mask-Guided Dual-Channel Neural Network for DAS-VSP Data Reconstruction},
      author={Bai, Zhuo and Zhao, Shu-Hong},
      journal={IEEE Sensors Journal},
      year={2026},
      publisher={IEEE}
    }

*(Note: Please update the citation with the exact volume, issue, and page numbers once assigned by the publisher.)*

## License

This project is licensed under the MIT License - see the LICENSE file for details. For academic and non-commercial use only.
