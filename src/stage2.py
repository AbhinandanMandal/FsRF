import torch.nn as nn
import torch
from sict import SICTPipeline
from dst import DSTPipeline
from decoder import CNNDecoder

class FsRF(nn.Module):
    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        self.sict_pipeline = SICTPipeline(config = config).to(device)
        self.dst_pipeline = DSTPipeline(config = config, device = device)
        self.decoder = CNNDecoder(
                C = config["content_feature_dim"],
                H = config["img_H"],
                W = config["img_W"],
                grid_size = config["grid_size"]).to(device)

    def forward(self, features, batch):

        # features.shape = [B, R, N, 256]
        self.sict_dict = self.sict_pipeline(features = features) # features.shape = [B, R, N, C]

        # DST BYPASS
        dst_dict = self.dst_pipeline(batch = batch, sict_dict = self.sict_dict) # Styled Features [B, R, C]
        F_c = dst_dict["F_c"]
        F_s = dst_dict["F_s"]
        F_cs = dst_dict["F_cs"]

        rgb_img = self.decoder(F_cs)
        
        return {
            "F_c": F_c,
            "F_s": F_s,
            "F_cs": F_cs, # Styled Features [B, R, C]
            "rgb_img": rgb_img # Rendered RGB Style Image (Content + Style)
        }