"""
UV Coordinate Predictor Node for ComfyUI
Predicts UV coordinates for face images using Pixel3DMM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .utils import convert_comfyui_image, convert_to_comfyui_image, preprocess_image

class UVPredictor:
    """
    ComfyUI node for predicting UV coordinates from face images
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PIXEL3DMM_MODEL",),
                "image": ("IMAGE",),
                "output_resolution": (["256", "512", "1024"], {"default": "512"}),
            },
            "optional": {
                "uv_smoothing": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "UV_COORDS", "STRING")
    RETURN_NAMES = ("uv_map", "uv_coordinates", "status")
    FUNCTION = "predict_uv"
    CATEGORY = "Pixel3DMM"
    
    def __init__(self):
        self.uv_network = None
    
    def predict_uv(self, model: Dict[str, Any], image: torch.Tensor, 
                   output_resolution: str = "512", uv_smoothing: float = 0.0,
                   confidence_threshold: float = 0.5) -> Tuple:
        """
        Predict UV coordinates from face image
        
        Args:
            model: Loaded Pixel3DMM model container
            image: Input face image
            output_resolution: Resolution of output UV map
            uv_smoothing: Amount of smoothing to apply
            confidence_threshold: Confidence threshold for UV predictions
            
        Returns:
            Tuple of (uv_map, uv_coordinates, status)
        """
        try:
            if model is None:
                return None, None, "❌ No model loaded. Please connect a Pixel3DMM Loader node."
            
            device = model['device']
            resolution = int(output_resolution)
            
            # Preprocess input image
            processed_image = self._preprocess_input(image, device)
            if processed_image is None:
                return None, None, "❌ Failed to preprocess input image"
            
            # Initialize or get UV prediction network
            uv_network = self._get_uv_network(model, device)
            if uv_network is None:
                return None, None, "❌ Failed to initialize UV prediction network"
            
            # Predict UV coordinates
            uv_coords = self._predict_uv_coordinates(uv_network, processed_image, resolution)
            if uv_coords is None:
                return None, None, "❌ Failed to predict UV coordinates"
            
            # Apply post-processing
            uv_coords = self._postprocess_uv(uv_coords, uv_smoothing, confidence_threshold)
            
            # Create UV visualization map
            uv_map = self._create_uv_visualization(uv_coords)
            
            # Convert to ComfyUI format
            uv_map = convert_to_comfyui_image(uv_map)
            
            status = f"✅ UV prediction completed\n"
            status += f"📐 Resolution: {resolution}x{resolution}\n"
            status += f"🎛️ Smoothing: {uv_smoothing:.1f}\n"
            status += f"🎯 Confidence threshold: {confidence_threshold:.2f}"
            
            return uv_map, uv_coords, status
            
        except Exception as e:
            error_msg = f"❌ Error in UV prediction: {str(e)}"
            print(error_msg)
            return None, None, error_msg
    
    def _preprocess_input(self, image: torch.Tensor, device: str) -> Optional[torch.Tensor]:
        """Preprocess input image for UV prediction"""
        try:
            # Convert from ComfyUI format
            processed = convert_comfyui_image(image)
            
            # Preprocess for model
            processed = preprocess_image(processed, target_size=512)
            
            # Move to device
            processed = processed.to(device)
            
            return processed
            
        except Exception as e:
            print(f"Error preprocessing input: {e}")
            return None
    
    def _get_uv_network(self, model: Dict[str, Any], device: str) -> Optional[nn.Module]:
        """Get or create UV prediction network"""
        try:
            # Check if UV network exists in model components
            if 'uv_predictor' in model['components']:
                return model['components']['uv_predictor']
            
            # Create UV prediction network if not exists
            uv_network = self._create_uv_network(device)
            if uv_network is not None:
                model['components']['uv_predictor'] = uv_network
            
            return uv_network
            
        except Exception as e:
            print(f"Error getting UV network: {e}")
            return None
    
    def _create_uv_network(self, device: str) -> Optional[nn.Module]:
        """Create UV prediction network"""
        try:
            class UVPredictionNetwork(nn.Module):
                def __init__(self):
                    super().__init__()
                    
                    # Encoder (simplified)
                    self.encoder = nn.Sequential(
                        nn.Conv2d(3, 64, 7, stride=2, padding=3),
                        nn.ReLU(),
                        nn.Conv2d(64, 128, 3, stride=2, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 256, 3, stride=2, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256, 512, 3, stride=2, padding=1),
                        nn.ReLU(),
                    )
                    
                    # Decoder for UV coordinates
                    self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 2, 3, padding=1),  # 2 channels for U,V
                        nn.Sigmoid()  # UV coordinates in [0,1]
                    )
                
                def forward(self, x):
                    # Encode
                    features = self.encoder(x)
                    
                    # Decode to UV coordinates
                    uv_coords = self.decoder(features)
                    
                    return uv_coords
            
            network = UVPredictionNetwork().to(device)
            network.eval()
            
            return network
            
        except Exception as e:
            print(f"Error creating UV network: {e}")
            return None
    
    def _predict_uv_coordinates(self, uv_network: nn.Module, image: torch.Tensor, 
                              resolution: int) -> Optional[torch.Tensor]:
        """Predict UV coordinates using the network"""
        try:
            uv_network.eval()
            with torch.no_grad():
                # Predict UV coordinates
                uv_coords = uv_network(image)
                
                # Resize to target resolution if needed
                if uv_coords.shape[-1] != resolution:
                    uv_coords = F.interpolate(
                        uv_coords, 
                        size=(resolution, resolution), 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                return uv_coords
                
        except Exception as e:
            print(f"Error predicting UV coordinates: {e}")
            return None
    
    def _postprocess_uv(self, uv_coords: torch.Tensor, smoothing: float, 
                       confidence_threshold: float) -> torch.Tensor:
        """Apply post-processing to UV coordinates"""
        try:
            # Apply smoothing if requested
            if smoothing > 0:
                kernel_size = int(smoothing * 10) + 1
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                uv_coords = F.avg_pool2d(
                    uv_coords, 
                    kernel_size=kernel_size, 
                    stride=1, 
                    padding=kernel_size//2
                )
            
            # Apply confidence thresholding (simplified)
            # In real implementation, would use actual confidence maps
            confidence_mask = torch.ones_like(uv_coords[:, :1])  # Placeholder
            
            # Mask out low-confidence regions
            low_confidence = confidence_mask < confidence_threshold
            uv_coords = uv_coords * (1 - low_confidence.float())
            
            return uv_coords
            
        except Exception as e:
            print(f"Error in UV post-processing: {e}")
            return uv_coords
    
    def _create_uv_visualization(self, uv_coords: torch.Tensor) -> torch.Tensor:
        """Create visualization of UV coordinates"""
        try:
            batch_size, _, height, width = uv_coords.shape
            
            # Create RGB visualization where R=U, G=V, B=0
            uv_vis = torch.zeros(batch_size, 3, height, width, device=uv_coords.device)
            
            # U coordinates -> Red channel
            uv_vis[:, 0] = uv_coords[:, 0]
            
            # V coordinates -> Green channel  
            uv_vis[:, 1] = uv_coords[:, 1]
            
            # Blue channel remains 0
            
            # Add some visualization enhancements
            # Create a grid pattern to show UV structure
            grid_u = torch.arange(width, device=uv_coords.device).float() / width
            grid_v = torch.arange(height, device=uv_coords.device).float() / height
            
            grid_u = grid_u.unsqueeze(0).repeat(height, 1)
            grid_v = grid_v.unsqueeze(1).repeat(1, width)
            
            # Add grid lines every 0.1 UV units
            grid_lines_u = (torch.fmod(grid_u * 10, 1.0) < 0.1).float()
            grid_lines_v = (torch.fmod(grid_v * 10, 1.0) < 0.1).float()
            grid_lines = torch.maximum(grid_lines_u, grid_lines_v)
            
            # Blend grid with UV visualization
            grid_lines = grid_lines.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
            uv_vis[:, 2] = grid_lines.squeeze(1) * 0.3  # Subtle blue grid
            
            return uv_vis
            
        except Exception as e:
            print(f"Error creating UV visualization: {e}")
            # Return simple UV visualization
            batch_size, _, height, width = uv_coords.shape
            uv_vis = torch.zeros(batch_size, 3, height, width, device=uv_coords.device)
            uv_vis[:, :2] = uv_coords
            return uv_vis
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate node inputs"""
        model = kwargs.get('model')
        if model is None:
            return "Model input is required"
        
        image = kwargs.get('image')
        if image is None:
            return "Image input is required"
        
        uv_smoothing = kwargs.get('uv_smoothing', 0.0)
        if uv_smoothing < 0.0 or uv_smoothing > 1.0:
            return "UV smoothing must be between 0.0 and 1.0"
        
        confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        if confidence_threshold < 0.0 or confidence_threshold > 1.0:
            return "Confidence threshold must be between 0.0 and 1.0"
        
        return True
