import torch.nn as nn


class UWBHead(nn.Module):
    """Task head for UWB coordinate or confidence prediction."""

    def __init__(self, in_dim=128, hidden_dims=None, task_dim=2, dropout=0.1, final_act=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = (max(in_dim // 2, 1),)

        dims = [in_dim] + list(hidden_dims) + [task_dim]
        layers = []
        for idx, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(input_dim, output_dim))
            if idx < len(dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

        if final_act is None or str(final_act).lower() == "none":
            self.final_act = nn.Identity()
        elif final_act == "relu":
            self.final_act = nn.ReLU(inplace=False)
        elif final_act == "sigmoid":
            self.final_act = nn.Sigmoid()
        elif final_act == "tanh":
            self.final_act = nn.Tanh()
        else:
            raise ValueError("final_act must be one of [None, 'relu', 'sigmoid', 'tanh']")

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.ndim == 3:
            if x.size(1) != 1:
                raise ValueError("If x is 3D, expected shape [B, 1, C]")
            x = x[:, 0, :]
        return self.final_act(self.mlp(x))


class UWBTokenHead(nn.Module):
    """Project the compact UWB feature to the visual token dimension."""

    def __init__(self, in_dim=128, token_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, token_dim),
            nn.LayerNorm(token_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.ndim == 3:
            if x.size(1) != 1:
                raise ValueError("If x is 3D, expected shape [B, 1, C]")
            x = x[:, 0, :]
        return self.mlp(x).unsqueeze(1)
