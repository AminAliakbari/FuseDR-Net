# -*- coding: utf-8 -*-
"""
FuseDR_Net Model Architecture for RUL Prediction.

This module defines the components and main architecture for the FuseDR-Net,
a deep learning model designed for time-series forecasting, particularly for
Remaining Useful Life (RUL) prediction of batteries. It uses a dual-branch
architecture to process trend and fluctuation components of the signal separately.
The trend branch uses Mamba blocks, while the fluctuation branch uses attention
and wavelet transforms.

Classes:
    CausalConv1d: A 1D convolutional layer that ensures causality.
    WaveletBlock: A block that applies 1D Discrete Wavelet Transform for signal processing.
    TemporalAttentionLayer: A standard multi-head self-attention layer.
    FuseDR_Net: The main dual-branch model architecture.
    FuseDR_Net_NetModel: A wrapper to make the model compatible with the pytorch-forecasting library.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

import pytorch_wavelets as pwt
from pytorch_forecasting.models import BaseModel
from ModelsModify.MambaSimple import ResidualBlock as MambaLayer, DataEmbedding

# ---------------------------- Model Components ----------------------------

class CausalConv1d(nn.Module):
    """A 1D convolutional layer that maintains causality.

    It pads the input on the left before convolution, ensuring that the output
    at timestep `t` only depends on inputs from timestep `t` and earlier.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        # Calculate padding to maintain sequence length and causality
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, L).

        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class WaveletBlock(nn.Module):
    """
    A block that processes a sequence using 1D Discrete Wavelet Transform (DWT).

    It decomposes the signal into approximation (cA) and detail (cD) coefficients,
    processes them with separate causal convolutions, and reconstructs the signal
    using Inverse DWT (IDWT). Includes a residual connection and layer normalization.
    """
    def __init__(self, d_model: int, wavelet: str = 'db4', level: int = 1):
        super().__init__()
        if level != 1:
            raise NotImplementedError("Only Level 1 DWT is currently supported.")
        
        self.dwt = pwt.DWT1D(wave=wavelet, J=level, mode='symmetric')
        self.idwt = pwt.IDWT1D(wave=wavelet, mode='symmetric')
        
        # Causal convolutions to process the wavelet coefficients
        self.cA_processor = CausalConv1d(in_channels=d_model, out_channels=d_model, kernel_size=3)
        self.cD_processor = CausalConv1d(in_channels=d_model, out_channels=d_model, kernel_size=3)
        
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D).

        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        residual = x
        # Permute to (B, D, L) for DWT
        x_permuted = x.permute(0, 2, 1)

        # Apply DWT
        cA, cD_list = self.dwt(x_permuted)
        cD = cD_list[0]

        # Process coefficients
        cA_processed = self.cA_processor(cA)
        cD_processed = self.cD_processor(cD)

        # Reconstruct signal using IDWT
        reconstructed_permuted = self.idwt((cA_processed, [cD_processed]))
        reconstructed_signal = reconstructed_permuted.permute(0, 2, 1)

        # Pad the reconstructed signal if its length is altered by DWT/IDWT
        if reconstructed_signal.shape[1] != residual.shape[1]:
            padding_size = residual.shape[1] - reconstructed_signal.shape[1]
            reconstructed_signal = F.pad(reconstructed_signal, (0, 0, 0, padding_size))

        # Add residual connection and apply layer normalization
        output = self.layer_norm(residual + reconstructed_signal)
        return output


class TemporalAttentionLayer(nn.Module):
    """A standard multi-head self-attention layer for processing temporal sequences.
    
    Includes a residual connection and layer normalization.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D).
            mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        residual = x
        B, L, _ = x.shape

        # Project and reshape for multi-head attention
        q = self.query(x).view(B, L, self.n_heads, -1).transpose(1, 2)
        k = self.key(x).view(B, L, self.n_heads, -1).transpose(1, 2)
        v = self.value(x).view(B, L, self.n_heads, -1).transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model // self.n_heads)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(self.dropout(attn_weights), v)
        
        # Concatenate heads and apply final projection
        attended = attended.transpose(1, 2).contiguous().view(B, L, -1)
        output = self.out_proj(attended)
        
        # Add residual connection and apply layer normalization
        return self.layer_norm(residual + output)

# ------------------------------- FuseDR_Net Decomposed Architecture ------------------------------

class FuseDR_Net(nn.Module):
    """The main FuseDR-Net model architecture for time-series forecasting.

    It decomposes input features into a trend component and a fluctuation
    component. The trend is processed by a Mamba-based branch, while 
    fluctuations are handled by a branch with attention and wavelet blocks.
    The outputs are fused using a dynamic gate for the final prediction.
    """
    def __init__(self, seq_len: int, pred_len: int, d_model: int, e_layers: int, n_heads: int, 
                 n_fluctuation_features: int, n_trend_features: int, ablation_mode: str = 'full'):
        super(FuseDR_Net, self).__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.ablation_mode = ablation_mode
        self.n_fluctuation_features = n_fluctuation_features
        self.n_trend_features = n_trend_features
        
        # --- Define Feature Slices for Readability and Maintenance ---
        self.CYCLE_SLICE = slice(0, 1)
        self.HEALTH_SLICE = slice(1, 17)
        self.TREND_SLICE = slice(17, 18)
        self.NOISE_SLICE = slice(18, 19)

        # --- Embeddings for each branch ---
        self.trend_embedding = DataEmbedding(c_in=self.n_trend_features, d_model=d_model, dropout=0.1)
        self.fluctuation_embedding = DataEmbedding(c_in=self.n_fluctuation_features, d_model=d_model, dropout=0.1)

        # --- Model Branches ---
        # Trend Branch (Mamba)
        d_inner = d_model * 2
        dt_rank = math.ceil(d_model / 16)
        d_ff = d_model * 2  # Define d_ff internally based on d_model
        self.trend_branch = nn.ModuleList([
            MambaLayer(d_inner=d_inner, dt_rank=dt_rank, d_model=d_model, d_ff=d_ff) for _ in range(e_layers)
        ])
        
        # Fluctuation Branch (Attention + Wavelet)
        self.fluctuation_branch = nn.ModuleList()
        for _ in range(e_layers):
            self.fluctuation_branch.append(TemporalAttentionLayer(d_model=d_model, n_heads=n_heads))
            self.fluctuation_branch.append(WaveletBlock(d_model=d_model, wavelet='db4')) 

        # --- Fusion and Projection ---
        # Dynamic gate for fusing the two branches
        self.gate_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        # Final projection layer to get the prediction
        self.projection = nn.Linear(d_model, 1)


    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FuseDR_Net model.

        Args:
            x_enc (torch.Tensor): The encoder input tensor of shape (B, L, D_features).

        Returns:
            torch.Tensor: The prediction tensor of shape (B, pred_len, 1).
        """
        # --- 1. Feature Separation ---
        # Use the predefined slices for clarity and easier maintenance.
        cycle_feature = x_enc[:, :, self.CYCLE_SLICE]
        health_features = x_enc[:, :, self.HEALTH_SLICE]
        trend_feature = x_enc[:, :, self.TREND_SLICE]
        fluctuation_noise_feature = x_enc[:, :, self.NOISE_SLICE]
        
        trend_input = torch.cat([trend_feature, cycle_feature], dim=-1)
        fluctuation_input = torch.cat([health_features, fluctuation_noise_feature, cycle_feature], dim=-1)

        # --- 2. Branch Processing (handles different ablation modes) ---
        # Trend Branch
        trend_output = 0.0
        if self.ablation_mode in ['full', 'trend_only']:
            trend_emb = self.trend_embedding(trend_input, x_mark=None)
            trend_output = trend_emb
            for layer in self.trend_branch:
                trend_output = layer(trend_output)

        # Fluctuation Branch
        fluctuation_output = 0.0
        if self.ablation_mode in ['full', 'fluctuation_only']:
            fluctuation_emb = self.fluctuation_embedding(fluctuation_input, x_mark=None)
            B, L, _ = fluctuation_emb.shape
            causal_mask = torch.triu(torch.ones(L, L, device=x_enc.device), diagonal=1).bool()
            fluctuation_output = fluctuation_emb
            for layer in self.fluctuation_branch:
                if isinstance(layer, TemporalAttentionLayer):
                    fluctuation_output = layer(fluctuation_output, mask=causal_mask)
                else:
                    fluctuation_output = layer(fluctuation_output)
        
        # --- 3. Fusion and Final Prediction ---
        # Select the final representation based on the ablation mode.
        if self.ablation_mode == 'full':
            # Fuse with a dynamic gate
            gate = self.gate_layer(trend_output)
            combined_representation = trend_output + (fluctuation_output * gate)
        elif self.ablation_mode == 'trend_only': # NOTE: Renamed for consistency
            combined_representation = trend_output
        elif self.ablation_mode == 'fluctuation_only': # NOTE: Renamed for consistency
            combined_representation = fluctuation_output
        else:
            raise ValueError(f"Unknown ablation_mode: {self.ablation_mode}")

        # Projection
        final_prediction = self.projection(combined_representation)
        
        # Slicing: Return only the prediction length part of the sequence
        return final_prediction[:, -self.pred_len:, :]

# -------------------- Wrapper Class for Pytorch-Forecasting -------------------
class FuseDR_Net_NetModel(BaseModel):
    """
    A wrapper class to make FuseDR_Net compatible with the pytorch-forecasting framework.
    
    This class handles the specific data dictionary format used by the framework's
    TimeSeriesDataSet and DataLoader. It extracts the necessary tensors, passes
    them to the core FuseDR_Net model, and formats the output.
    """
    def __init__(self,
                 seq_len: int,
                 pred_len: int,
                 d_model: int,
                 e_layers: int,
                 n_heads: int,
                 n_fluctuation_features: int,
                 n_trend_features: int,
                 ablation_mode: str = 'full',
                 **kwargs):
        
        self.save_hyperparameters()
        super().__init__(**kwargs)

        # Instantiate the core network
        self.network = FuseDR_Net(
            seq_len=self.hparams.seq_len,
            pred_len=self.hparams.pred_len,
            d_model=self.hparams.d_model,
            e_layers=self.hparams.e_layers,
            n_heads=self.hparams.n_heads,
            n_fluctuation_features=self.hparams.n_fluctuation_features,
            n_trend_features=self.hparams.n_trend_features,
            ablation_mode=self.hparams.ablation_mode
        )
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Defines the forward pass for the model within the pytorch-forecasting framework.

        Args:
            x (Dict[str, torch.Tensor]): The input dictionary from the DataLoader.
                                         It must contain 'encoder_cont'.

        Returns:
            Dict[str, torch.Tensor]: An output dictionary with the 'prediction' key.
        """
        # Extract features from the input dictionary.
        # The target variable is typically the last column in 'encoder_cont', so we exclude it.
        input_data = x["encoder_cont"][:, :, :-1]
        
        # Get prediction from the core network
        prediction = self.network(x_enc=input_data)
        
        # Rescale the output to the original data scale
        prediction = self.transform_output(prediction, target_scale=x["target_scale"])
        
        # Format the output for the framework
        return self.to_network_output(prediction=prediction)