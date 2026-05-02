import torch
import torch.nn as nn


class UWBMLPEncoder(nn.Module):
    """Encode a fixed UWB window by flattening [T, 2] coordinates."""

    def __init__(self, in_dim=2, seq_len=10, hidden_dims=(128, 128), out_dim=128, dropout=0.1):
        super().__init__()
        self.in_dim = int(in_dim)
        self.seq_len = int(seq_len)
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")

        dims = [self.in_dim * self.seq_len] + list(hidden_dims) + [out_dim]
        layers = []
        for idx, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(input_dim, output_dim))
            if idx < len(dims) - 2:
                layers.append(nn.LayerNorm(output_dim))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, uwb_uv):
        if uwb_uv.ndim != 3 or uwb_uv.size(-1) != self.in_dim:
            raise ValueError("uwb_uv must have shape [B, T, in_dim]")

        if uwb_uv.shape[1] < self.seq_len:
            pad = uwb_uv[:, :1, :].repeat(1, self.seq_len - uwb_uv.shape[1], 1)
            uwb_uv = torch.cat([pad, uwb_uv], dim=1)
        uwb_uv = uwb_uv[:, -self.seq_len:, :]
        x = uwb_uv.reshape(uwb_uv.shape[0], self.seq_len * self.in_dim)
        return self.net(x).unsqueeze(1)


class UWBGRUEncoder(nn.Module):
    """Encode a UWB sequence with a GRU backbone."""

    def __init__(self, in_dim=2, input_proj_dim=64, hidden_dim=128, out_dim=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, input_proj_dim)
        self.gru = nn.GRU(input_proj_dim, hidden_dim, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, uwb_uv):
        if uwb_uv.ndim != 3 or uwb_uv.size(-1) != 2:
            raise ValueError("uwb_uv must have shape [B, T, 2]")

        x = torch.relu(self.input_proj(uwb_uv))
        y, _ = self.gru(x)
        return self.out_proj(y[:, -1, :]).unsqueeze(1)


class UWBDilatedTCNBlock(nn.Module):
    """Residual same-length dilated Conv1d block."""

    def __init__(self, channels=64, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class UWBTCNEncoder(nn.Module):
    """Encode a UWB sequence with a residual dilated TCN."""

    def __init__(self, in_dim=2, channels=64, dilations=(1, 2, 4), out_dim=128, kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_dim, channels, kernel_size=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                UWBDilatedTCNBlock(channels=channels, kernel_size=kernel_size, dilation=dilation, dropout=dropout)
                for dilation in dilations
            ]
        )
        self.proj = nn.Sequential(
            nn.Linear(channels, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, uwb_uv):
        if uwb_uv.ndim != 3 or uwb_uv.size(-1) != 2:
            raise ValueError("uwb_uv must have shape [B, T, 2]")

        x = self.input_conv(uwb_uv.transpose(1, 2))
        x = self.blocks(x).transpose(1, 2)
        return self.proj(x[:, -1, :]).unsqueeze(1)


UWBConv1DEncoder = UWBTCNEncoder
