import torch
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    """
    > This implements the original MLP Design discussed in the section-3:  Neural Radiance Field Scene Representation
    > The MLP is 1st Queried for all Encoded Sampled Points and then we add directions to these feature representation to compute the RGB values for each points
    > To avoid hardcoded design i'm passing position_dim and direction_dim explicitly.
    """
    def __init__(self, position_dim: int, hidden_dim: int = 256, content_feature_dim: int = 256):
        super().__init__()
        self.content_feature_dim = content_feature_dim

        self.block1 = nn.Sequential(
            nn.Linear(position_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim + position_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim), # Final feature layer (no ReLU)
        )

        self.sigma_layer = nn.Linear(hidden_dim, 1)

        self.content_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.content_feature_dim),
        )

        self.apply(self.init_weights)

    def forward(self, coordinates):
        # coordinates ---> Positional Encoded co-ordinates
        # directions ---> Positional Encoded directions

        # first processes the input 3D coordinate x with 8 fully-connected layers
        h = self.block1(coordinates) # (B, R, N, 256)

        # Skip connection: The MLP uses ReLU activations and one skip connection that concatenates the input to the fifth layer
        h = torch.cat([h, coordinates], dim=-1)
        feature = self.block2(h)

        # Get the Densities
        sigma = self.sigma_layer(feature)
        sigma = F.softplus(sigma) # convert this 256-dim feature vector to 1-dim volumne density (shape = (B, R, N, 1))

        # Get the Content Features (each 3D-point in space now have it's own content_features vector)
        # content_features = self.content_head(torch.cat([feature, directions], dim = -1)) # removing the dependency of the direction from the MLP because we don't want view-dependency RGB anymore
        content_features = self.content_head(feature)
        return sigma, content_features


    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
