"""Additional competitive baselines for publication experiments."""
from __future__ import annotations

import torch
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, chomp): super().__init__(); self.chomp = chomp
    def forward(self, x): return x[:, :, :-self.chomp] if self.chomp else x


class TemporalBlock(nn.Module):
    def __init__(self, cin, cout, kernel, dilation, dropout=0.1):
        super().__init__()
        pad=(kernel-1)*dilation
        self.net=nn.Sequential(
            nn.Conv1d(cin,cout,kernel,padding=pad,dilation=dilation), Chomp1d(pad), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(cout,cout,kernel,padding=pad,dilation=dilation), Chomp1d(pad), nn.GELU(), nn.Dropout(dropout))
        self.res=nn.Conv1d(cin,cout,1) if cin!=cout else nn.Identity()
        self.norm=nn.LayerNorm(cout)
    def forward(self,x):
        y=self.net(x)+self.res(x)
        return self.norm(y.transpose(1,2)).transpose(1,2)


class TCNForecaster(nn.Module):
    def __init__(self,n_features,horizon,channels=(32,64,64),kernel=3):
        super().__init__(); layers=[]; cin=n_features
        for i,cout in enumerate(channels):
            layers.append(TemporalBlock(cin,cout,kernel,2**i)); cin=cout
        self.tcn=nn.Sequential(*layers); self.head=nn.Linear(cin,horizon)
    def forward(self,x):
        z=self.tcn(x.transpose(1,2)); return self.head(z[:,:,-1])


class MambaForecaster(nn.Module):
    """Optional Mamba baseline using mamba-ssm when available."""
    def __init__(self,n_features,horizon,d_model=64,layers=2,d_state=16,d_conv=4,expand=2):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise ImportError("MambaForecaster requires optional dependency `mamba-ssm`.") from exc
        self.input=nn.Linear(n_features,d_model)
        self.layers=nn.ModuleList([Mamba(d_model=d_model,d_state=d_state,d_conv=d_conv,expand=expand) for _ in range(layers)])
        self.norm=nn.LayerNorm(d_model); self.head=nn.Linear(d_model,horizon)
    def forward(self,x):
        z=self.input(x)
        for layer in self.layers: z=z+layer(z)
        return self.head(self.norm(z)[:,-1])
