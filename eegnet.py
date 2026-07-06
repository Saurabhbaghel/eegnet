from typing import Tuple, Literal

import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ["EEGNet"]

class EEGNetBlock1(nn.Module):
    def __init__(self, 
                 f1: int, 
                 depth_mul: int, 
                 n_channels: int,
                 drop: float=0.5):
        super().__init__()
        self.f1_conv2d = nn.Conv2d(kernel_size=(1, 64), # acc. to paper
                                   in_channels=1,
                                   out_channels=f1, # acc. to paper
                                   padding=(0, 32),  # to match author's keras code 
                                   bias=False
                                   ) 
        
        
        self.batchnorm1 = nn.BatchNorm2d(f1)
        
        self.depthwise_spatial_conv2d = nn.Conv2d(in_channels=f1, 
                                                  out_channels=f1*depth_mul,    # multiplied by depth
                                                  kernel_size=(n_channels, 1),
                                                  groups=f1,
                                                  bias=False
                                                  )
        
        self.batchnorm2 = nn.BatchNorm2d(f1*depth_mul)
        
        self.avg_pool2d = nn.AvgPool2d((1, 4))  # acc. to paper
        
        self.dropout = nn.Dropout(p=drop)    # acc. to paper
        
    def forward(self, x: torch.Tensor):
        
        # 1. reshape
        # x = x.unsqueeze(1)  # (1, C, T)
        
        # 2. Linear activation(conv2d)
        x = self.f1_conv2d(x)   # (f1, C, T)
        
        # 3. batchnorm
        x = self.batchnorm1(x)   # (f1, C, T)
        
        # 4. depthwise conv2d
        x = self.depthwise_spatial_conv2d(x)    # (f1*depth_mul, 1, T)
        
        # 5. batchnorm
        x = self.batchnorm2(x)  # (f1*depth_mul, 1, T) 
        
        # 6. elu activation
        x = F.elu(x)
        
        # 7. avg pooling
        x = self.avg_pool2d(x)  # (f1*depth_mul, 1, T//4)
        
        # 8. dropout
        x = self.dropout(x)     # (f1*depth_mul, 1, T//4)
        
        return x
        
        
class EEGNetBlock2(nn.Module):
    def __init__(self, 
                 n_classes: int,
                 input_shape: Tuple,
                 f2: int,   # first separable conv 2d out channels 
                 drop: float=0.5):
        super().__init__()
        channels, _, tps = input_shape
        self._sep_conv2d_temporal = nn.Conv2d(in_channels=channels,
                                            out_channels=f2,
                                            kernel_size=(1, 16),
                                            groups=channels,
                                            bias=False,
                                            padding="same") # to retain shape
        
        self._sep_conv2d_pointwise = nn.Conv2d(in_channels=channels,
                                              out_channels=f2,
                                              kernel_size=1, 
                                              bias=False)
        
        self.batchnorm = nn.BatchNorm2d(f2)

        self.avg_pool2d = nn.AvgPool2d((1, 8))  # acc. paper
        
        self.dropout = nn.Dropout(p=drop)
        
        self.flatten = nn.Flatten()
        
        self.dense = nn.Linear(in_features=f2*(tps // 8), out_features=n_classes, bias=False)

    def depthwise_separable(self, x: torch.Tensor):
        x = self._sep_conv2d_temporal(x)    # 
        x = self._sep_conv2d_pointwise(x)
        return x

    def forward(self, x: torch.Tensor):
        # 1. separable conv
        x = self.depthwise_separable(x) # 
        
        # 2. batchnorm
        x = self.batchnorm(x)
        
        # 3. activation
        x = F.elu(x)
        
        # 4. avg pool
        x = self.avg_pool2d(x)
        
        # 5. dropout
        x = self.dropout(x)
        
        # 6. flatten
        x = self.flatten(x)
        
        # 7. dense
        x = self.dense(x)
        
        return x
        
class EEGNet(nn.Module):
    """
    Implemented Paper
    
    For training - 
    Vary: f1, D
    
    `f2 = D * f1`  
    But, in principle it can take any value. 
    `f2 < (>) D * f1` denotes a compressed (overcomplete) representation, 
    learning a fewer (more) feature maps than inputs.
    
    """
    def __init__(self, 
                 f1: int, 
                 depth_mul: int, 
                 n_channels: int,   # C or dim 0
                 tps: int,  # T or time points or dim 1
                 n_classes: int,
                 f2: int=None,
                 classification_type: Literal["within-subject", "cross-subject"] = "within-subject"
                 ):
        super().__init__()
        self.block1 = EEGNetBlock1(f1, 
                                   depth_mul, 
                                   n_channels, 
                                   0.5 if classification_type == "within-subject" else 0.25)
        self.block2 = EEGNetBlock2(n_classes, 
                                   (depth_mul * f1, 1, tps // 4),
                                   f2 if f2 else f1*depth_mul,    # f2, acc. to paper
                                   0.5 if classification_type == "within-subject" else 0.25)
        
    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        
        return x
        
