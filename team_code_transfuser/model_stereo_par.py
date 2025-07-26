from utils import *
from transfuser_stereo_par import TransfuserBackbone
from latentTF_stereo import latentTFBackbone

# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

class LidarCenterNet(nn.Module):
    """
    Encoder network for LiDAR and image input, predicts acceleration.
    Args:
        config: Configuration object with model settings.
        device: Torch device (CPU or CUDA).
        backbone: Backbone model type for feature extraction.
        image_architecture: Backbone architecture for image processing.
        lidar_architecture: Backbone architecture for LiDAR processing.
        use_velocity: Whether to include velocity in the model input.
    """
    def __init__(self, config, device, backbone, image_architecture='resnet34', point_cloud_architecture='resnet18', use_velocity=1):
        super().__init__()
        self.device = device
        self.config = config
        self.use_velocity = use_velocity

        # Backbone for feature extraction
        if backbone == 'transFuser':
            self.backbone = TransfuserBackbone(config, image_architecture, point_cloud_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'latentTF'):
            self.backbone = latentTFBackbone(config, image_architecture, point_cloud_architecture, use_velocity=use_velocity).to(self.device)
        
        else:
            raise ValueError("Unsupported backbone type. Use 'transFuser'or 'latentTF'.")

        # Prediction head for acceleration
        self.head = nn.Sequential(
            nn.Linear(self.backbone.fused_features, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output: acceleration
        ).to(self.device)

    def forward(self, image, lidar_bev, velocity=None):
        """
        Forward pass through the model.
        Args:
            image: Input RGB image tensor (B, C, H, W).
            lidar_bev: Input LiDAR BEV tensor (B, C, H, W).
            velocity: Current velocity tensor (B, 1) (optional).
        Returns:
            Acceleration prediction tensor (B, 1).
        """
        # Extract features
        fused_features = self.backbone(image, lidar_bev, velocity)
        # fused_features = output_features[-1]

        # Flatten features for the head
        # fused_features = fused_features.view(fused_features.size(0), -1)

        # Predict acceleration
        acceleration = self.head(fused_features)

        return acceleration
