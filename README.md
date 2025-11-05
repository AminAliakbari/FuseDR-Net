## FuseDR-Net: a fusion-based network for battery remaining useful life prediction using causally decomposed health indicators

This repository contains the codes for our paper, published in Energy Conversion and Managment: X

DOI: https://doi.org/10.1016/j.ecmx.2025.101341

## Abstract

Accurate prediction of the remaining useful life (RUL) of lithium-ion batteries can guarantee the safety and reliability of energy systems. Yet, this task is problematic owing to complex degradation mechanisms, particularly regarding the capacity regeneration phenomenon. To overcome these issues, this paper introduces a Fusion-based Degradation-Regeneration Network (FuseDR-Net), a novel prognostic model especially conceived to solve the problem. The model utilizes a classical causal running-minimum feature engineering method to decompose the capacity signal into a strictly non-increasing degradation trend and a fluctuating regeneration signal. This decomposition enables a specialized dual-branch architecture to model these distinct dynamics in parallel. The trend branch utilizes MambaLayers to effectively capture long-term dependencies, while a window-causal wavelet-attention fluctuation branch responds to transient regeneration dynamics in conjunction with several health indicators. Compared to a set of state-of-the-art (SOTA) models on the publicly released NASA and TJU datasets, FuseDR-Net demonstrates improved performance and generalization capability. In the challenging NASA dataset, our model achieves a terminal RUL prediction with an average absolute error (AAE) of only 1.0 cycle, representing a 16.7% increase in accuracy over the next-best baseline model. It also excels at curve-fitting with an average mean absolute error (AMAE) of 0.0079. Additionally, the model achieves rapid inference time (under 0.23 s on the NASA dataset and about 0.57 s on the larger TJU dataset), confirming its computational efficiency and viable application in real-time battery management systems (BMS). The results confirm that FuseDR-Net, by modeling the decomposed competing signals inherent in battery aging, provides a more robust, accurate, and computationally efficient solution to battery prognostics.

## Requirements

The python version 3.12.8 was used for this project.
```bash
torch==2.5.1+cu124
einops==0.8.1
lightning==2.5.1.post0
matplotlib==3.10.0
numpy==1.26.3
pandas==2.2.3
pytorch-forecasting==1.3.0
pytorch-wavelets==1.3.0
PyYAML==6.0.2
reformer-pytorch==1.4.4
scikit-learn==1.6.1
scipy==1.15.2
torchmetrics==1.7.1
```

## Datasets

The raw NASA Battery Dataset can be obtained from this URL:

https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

The URL of preprocessed, ready-to-use TJU dataset is as follows:

TJU dataset: https://github.com/wang-fujin/PINN4SOH/tree/main/data/TJU%20data/Dataset_3_NCM_NCA_battery.

Note 1: The raw TJU dataset can be obtained from the Zenodo database under accession code: https://doi.org/10.5281/zenodo.6379165

## Prepration

1- place the datasets in the following pathes: 

    NASA: .\data\NASA data\data
    
    TJU: .\data\TJU data\Dataset_3_NCM_NCA_battery


2- dataset prepration:

    - For NASA dataset:
    
          1- Run "NASA_Feature_Extraction.py" to extract needed battery HIs.
          
          2- Run "NASADataPreProcess.py" to create numpy version of the dataset.

    - For TJU dataset:
    
          1- Thanks to (Wang et al. 2024) ready-to-use TJU dataset can be obtained from the mentioned URL.
          
          2- RUN "TJUDataPreProcess.py" to create numpy version of the dataset.

          
3- Here is a critical note: The training scripts has been tuned for the NASA dataset. Therefore, to generate results on TJU dataset, make sure to set the args. correctly and change the following lines:

        from NASADataPreProcess import MultiVariateBatteryDataProcess ---> from TJUDataPreProcess import MultiVariateBatteryDataProcess
        
        BatteryData = np.load('data/NASA data/NASA_dataset.npy', allow_pickle=True).item() ---> BatteryData = np.load('data/TJU data/TJU_dataset.npy', allow_pickle=True).item()
    
    Note: The SPs must be set correctly in the whole script. for NASA it must be (50, 70, 90) and for TJU it must be (200, 300, 400) to regenerate the results of the paper.


4-- Modify the "Helper_Plot.py" to suit each dataset.

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

Work and Code are inspired by https://github.com/USTC-AI4EEE/RUL-Mamba.

## Citation

If you find our work useful in your research, please consider citing:

```latex
@article{َALIAKBARI2025101341,
    title = {FuseDR-Net: a fusion-based network for battery remaining useful life prediction using causally decomposed health indicators},
    journal = {Energy Conversion and Managment : X},
    volume = {28},
    pages = {101341},
    year = {2025},
    issn = {2590-1745},
    doi = {https://doi.org/10.1016/j.ecmx.2025.101341},
    url = {https://www.sciencedirect.com/science/article/pii/S2590174525004738},
    author = {Amin Aliakbari and Masoud Masih-Tehrani},
}
```
## License

This code is released under the MIT license, see LICENSE for details.

## Contact info

If you had any difficaluties or questions feel free to cantact me via amin_aliakbari@alumni.iust.ac.ir
