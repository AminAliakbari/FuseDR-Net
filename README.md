## FuseDR-Net: a fusion-based network for battery remaining useful life prediction using causally decomposed health indicators

This repository contains the codes for our paper, published in Energy Conversion and Managment: X

DOI: https://doi.org/10.1016/j.ecmx.2025.101341

## Abstract

Accurate prediction of the remaining useful life (RUL) of lithium-ion batteries can guarantee the safety and reliability of energy systems. Yet, this task is problematic owing to complex degradation mechanisms, particularly regarding the capacity regeneration phenomenon. To overcome these issues, this paper introduces a Fusion-based Degradation-Regeneration Network (FuseDR-Net), a novel prognostic model especially conceived to solve the problem. The model utilizes a classical causal running-minimum feature engineering method to decompose the capacity signal into a strictly non-increasing degradation trend and a fluctuating regeneration signal. This decomposition enables a specialized dual-branch architecture to model these distinct dynamics in parallel. The trend branch utilizes MambaLayers to effectively capture long-term dependencies, while a window-causal wavelet-attention fluctuation branch responds to transient regeneration dynamics in conjunction with several health indicators. Compared to a set of state-of-the-art (SOTA) models on the publicly released NASA and TJU datasets, FuseDR-Net demonstrates improved performance and generalization capability. In the challenging NASA dataset, our model achieves a terminal RUL prediction with an average absolute error (AAE) of only 1.0 cycle, representing a 16.7% increase in accuracy over the next-best baseline model. It also excels at curve-fitting with an average mean absolute error (AMAE) of 0.0079. Additionally, the model achieves rapid inference time (under 0.23 s on the NASA dataset and about 0.57 s on the larger TJU dataset), confirming its computational efficiency and viable application in real-time battery management systems (BMS). The results confirm that FuseDR-Net, by modeling the decomposed competing signals inherent in battery aging, provides a more robust, accurate, and computationally efficient solution to battery prognostics.

## Requirements

The version of python is 3.12.8 .
```bash
numpy==1.21.6
numba==0.55.1
matplotlib==3.3.4
scipy==1.8.0
statsmodels==0.13.5
pytorch-lightning==1.9.5
pytorch-forecasting==0.10.3
sympy==1.12.1
reformer_pytorch==1.4.4
openpyxl==3.1.5
einops==0.8.0
```

## 3.Datasets

The raw NASA Battery Dataset can be obtained from this URL:

https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

The URL of preprocessed, ready-to-use TJU dataset is as follows:

TJU dataset: https://github.com/wang-fujin/PINN4SOH/tree/main/data/TJU%20data/Dataset_3_NCM_NCA_battery.

notation: The raw TJU dataset can be obtained from the Zenodo database under accession code: https://doi.org/10.5281/zenodo.6379165

## Training and Ecaluation

- an example for train and evaluate a new model：

```bash
python Decomposed_HIs_RUL_Prediction_FuseDR_Net.py
```

- Here is the expected output for one run:
    
```bash

Automatically selected GPU: 0

----- Processing Start Point: 50 for Battery: B0005 -----
Model: FuseDR-Net_full
Battery: B0005, Start Point: 50

Using default `ModelCheckpoint`. Consider installing `litmodels` package to enable `LitModelCheckpoint` for automatic upload to the Lightning model registry.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
You are using a CUDA device ('NVIDIA GeForce RTX 4050 Laptop GPU') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name            | Type       | Params | Mode
-------------------------------------------------------
0 | loss            | SMAPE      | 0      | train
1 | logging_metrics | ModuleList | 0      | train
2 | network         | FuseDR_Net | 78.9 K | train
-------------------------------------------------------
71.6 K    Trainable params
7.3 K     Non-trainable params
78.9 K    Total params
0.315     Total estimated model params size (MB)
60        Modules in train mode
0         Modules in eval mode
Epoch 32: 100%|███████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  7.16it/s, train_loss_step=0.00458, val_loss=0.0118, train_loss_epoch=0.00431] 
Using default `ModelCheckpoint`. Consider installing `litmodels` package to enable `LitModelCheckpoint` for automatic upload to the Lightning model registry.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Run 1: MAE=0.0070, RMSE=0.0104, R2=0.9939, RUL_real=73, RUL_pred=72, AE=1, RE=0.0139, Epochs=33, TrainTime=21.95s, InferTime=0.37s
Run 1: MAE=0.0070, RMSE=0.0104, R2=0.9939, RUL_real=73, RUL_pred=72, AE=1, RE=0.0139, Epochs=33, TrainTime=21.95s, InferTime=0.37s

```

## Acknowledgments

Work and Code is inspired by https://github.com/USTC-AI4EEE/RUL-Mamba.

## Citation

If you find our work useful in your research, please consider citing:

```latex
@article{َALIAKBARI2025101341,
    title = {FuseDR-Net: a fusion-based network for battery remaining useful life prediction using causally decomposed health indicators},
    journal = {Energy Conversion and Managment : X},
    volume = {28},
    pages = {101341},
    year = {2025},
    issn = {2352-152X},
    doi = {https://doi.org/10.1016/j.ecmx.2025.101341},
    url = {https://www.sciencedirect.com/science/article/pii/S2590174525004738},
    author = {Amin Aliakbari and Masoud Masih-Tehrani},
}
```
## License

his code is released under the MIT license, see LICENSE.md for details.


If you had any difficaluties or questions feel free to cantact me via amin_aliakbari@alumni.iust.ac.ir
