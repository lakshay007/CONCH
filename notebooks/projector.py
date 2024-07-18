import torch
import torch.nn as nn

class ImageEmbeddingProjector(nn.Module):
    def __init__(self, input_dim=384, output_dim=512, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, output_dim))
            else:
                self.layers.append(nn.Linear(output_dim, output_dim))
            if i < num_layers - 1:
                self.layers.append(nn.ReLU())
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


projector = ImageEmbeddingProjector()

projector = projector.to(device)