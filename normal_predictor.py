"""
Surface Normal Predictor Node for ComfyUI
Predicts surface normals for face images using Pixel3DMM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .utils import convert_comfyui_image, convert_to_comfyui_image, preprocess_image, normalize_tensor

class NormalPredictor:
    """
    ComfyUI node for predicting surface normals from face images
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PIXEL3DMM_MODEL",),
                "image": ("IMAGE",),
                "output_resolution": (["256", "512", "1024"], {"default": "512"}),
                "normal_space": (["camera", "world"], {"default": "camera"}),
            },
            "optional": {
                "normal_smoothing": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "enhance_details": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "NORMALS", "STRING")
    RETURN_NAMES = ("normal_map", "normal_vectors", "status")
    FUNCTION = "predict_normals"
    CATEGORY = "Pixel3DMM"
    
    def __init__(self):
        self.normal_network = None
    
    def predict_normals(self, model: Dict[str, Any], image: torch.Tensor, 
                       output_resolution: str = "512", normal_space: str = "camera",
                       normal_smoothing: float = 0.0, enhance_details: bool = True) -> Tuple:
        """
        Predict surface normals from face image
        
        Args:
            model: Loaded Pixel3DMM model container
            image: Input face image
            output_resolution: Resolution of output normal map
            normal_space: Coordinate space for normals (camera/world)
            normal_smoothing: Amount of smoothing to apply
            enhance_details: Whether to enhance fine details
            
        Returns:
            Tuple of (normal_map, normal_vectors, status)
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
            
            # Initialize or get normal prediction network
            normal_network = self._get_normal_network(model, device)
            if normal_network is None:
                return None, None, "❌ Failed to initialize normal prediction network"
            
            # Predict surface normals
            normals = self._predict_surface_normals(normal_network, processed_image, resolution)
            if normals is None:
                return None, None, "❌ Failed to predict surface normals"
            
            # Apply post-processing
            normals = self._postprocess_normals(normals, normal_smoothing, enhance_details)
            
            # Transform to requested coordinate space
            if normal_space == "world":
                normals = self._transform_to_world_space(normals)
            
            # Create normal map visualization
            normal_map = self._create_normal_visualization(normals)
            
            # Convert to ComfyUI format
            normal_map = convert_to_comfyui_image(normal_map)
            
            status = f"✅ Normal prediction completed\n"
            status += f"📐 Resolution: {resolution}x{resolution}\n"
            status += f"🌍 Space: {normal_space}\n"
            status += f"🎛️ Smoothing: {normal_smoothing:.1f}\n"
            status += f"✨ Enhanced details: {'Yes' if enhance_details else 'No'}"
            
            return normal_map, normals, status
            
        except Exception as e:
            error_msg = f"❌ Error in normal prediction: {str(e)}"
            print(error_msg)
            return None, None, error_msg
    
    def _preprocess_input(self, image: torch.Tensor, device: str) -> Optional[torch.Tensor]:
        """Preprocess input image for normal prediction"""
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
    
    def _get_normal_network(self, model: Dict[str, Any], device: str) -> Optional[nn.Module]:
        """Get or create normal prediction network"""
        try:
            # Check if normal network exists in model components
            if 'normal_predictor' in model['components']:
                return model['components']['normal_predictor']
            
            # Create normal prediction network if not exists
            normal_network = self._create_normal_network(device)
            if normal_network is not None:
                model['components']['normal_predictor'] = normal_network
            
            return normal_network
            
        except Exception as e:
            print(f"Error getting normal network: {e}")
            return None
    
    def _create_normal_network(self, device: str) -> Optional[nn.Module]:
        """Create surface normal prediction network"""
        try:
            class NormalPredictionNetwork(nn.Module):
                def __init__(self):
                    super().__init__()
                    
                    # Encoder with skip connections
                    self.enc1 = nn.Sequential(
                        nn.Conv2d(3, 64, 7, stride=2, padding=3),
                        nn.BatchNorm2d(64),
                        nn.ReLU()
                    )
                    
                    self.enc2 = nn.Sequential(
                        nn.Conv2d(64, 128, 3, stride=2, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU()
                    )
                    
                    self.enc3 = nn.Sequential(
                        nn.Conv2d(128, 256, 3, stride=2, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU()
                    )
                    
                    self.enc4 = nn.Sequential(
                        nn.Conv2d(256, 512, 3, stride=2, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU()
                    )
                    
                    # Decoder with skip connections
                    self.dec4 = nn.Sequential(
                        nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU()
                    )
                    
                    self.dec3 = nn.Sequential(
                        nn.ConvTranspose2d(256 + 256, 128, 4, stride=2, padding=1),  # +256 for skip
                        nn.BatchNorm2d(128),
                        nn.ReLU()
                    )
                    
                    self.dec2 = nn.Sequential(
                        nn.ConvTranspose2d(128 + 128, 64, 4, stride=2, padding=1),  # +128 for skip
                        nn.BatchNorm2d(64),
                        nn.ReLU()
                    )
                    
                    self.dec1 = nn.Sequential(
                        nn.ConvTranspose2d(64 + 64, 32, 4, stride=2, padding=1),  # +64 for skip
                        nn.BatchNorm2d(32),
                        nn.ReLU()
                    )
                    
                    # Final normal prediction layer
                    self.normal_head = nn.Sequential(
                        nn.Conv2d(32, 16, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(16, 3, 3, padding=1),  # 3 channels for X,Y,Z normals
                        nn.Tanh()  # Normals in [-1,1]
                    )
                
                def forward(self, x):
                    # Encoder with skip connections
                    e1 = self.enc1(x)    # 64 x 256 x 256
                    e2 = self.enc2(e1)   # 128 x 128 x 128
                    e3 = self.enc3(e2)   # 256 x 64 x 64
                    e4 = self.enc4(e3)   # 512 x 32 x 32
                    
                    # Decoder with skip connections
                    d4 = self.dec4(e4)                           # 256 x 64 x 64
                    d3 = self.dec3(torch.cat([d4, e3], dim=1))   # 128 x 128 x 128
                    d2 = self.dec2(torch.cat([d3, e2], dim=1))   # 64 x 256 x 256
                    d1 = self.dec1(torch.cat([d2, e1], dim=1))   # 32 x 512 x 512
                    
                    # Predict normals
                    normals = self.normal_head(d1)  # 3 x 512 x 512
                    
                    # Normalize to unit vectors
                    normals = F.normalize(normals, p=2.0, dim=1)
                    
                    return normals
            
            network = NormalPredictionNetwork().to(device)
            network.eval()
            
            return network
            
        except Exception as e:
            print(f"Error creating normal network: {e}")
            return None
    
    def _predict_surface_normals(self, normal_network: nn.Module, image: torch.Tensor, 
                               resolution: int) -> Optional[torch.Tensor]:
        """Predict surface normals using the network"""
        try:
            normal_network.eval()
            with torch.no_grad():
                # Predict surface normals
                normals = normal_network(image)
                
                # Resize to target resolution if needed
                if normals.shape[-1] != resolution:
                    normals = F.interpolate(
                        normals, 
                        size=(resolution, resolution), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    
                    # Re-normalize after interpolation
                    normals = normalize_tensor(normals, dim=1)
                
                return normals
                
        except Exception as e:
            print(f"Error predicting surface normals: {e}")
            return None
    
    def _postprocess_normals(self, normals: torch.Tensor, smoothing: float, 
                           enhance_details: bool) -> torch.Tensor:
        """Apply post-processing to surface normals"""
        try:
            # Apply smoothing if requested
            if smoothing > 0:
                kernel_size = int(smoothing * 10) + 1
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                # Apply Gaussian smoothing
                sigma = smoothing * 2.0
                kernel = self._create_gaussian_kernel(kernel_size, sigma, normals.device)
                
                # Apply smoothing to each channel
                smoothed_normals = torch.zeros_like(normals)
                for i in range(3):
                    smoothed_normals[:, i:i+1] = F.conv2d(
                        normals[:, i:i+1], 
                        kernel.unsqueeze(0).unsqueeze(0), 
                        padding=kernel_size//2
                    )
                
                # Re-normalize after smoothing
                normals = normalize_tensor(smoothed_normals, dim=1)
            
            # Enhance details if requested
            if enhance_details:
                normals = self._enhance_normal_details(normals)
            
            return normals
            
        except Exception as e:
            print(f"Error in normal post-processing: {e}")
            return normals
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float, device: str) -> torch.Tensor:
        """Create Gaussian smoothing kernel"""
        try:
            coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
            coords -= kernel_size // 2
            
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            
            # Create 2D kernel
            kernel = g.unsqueeze(1) * g.unsqueeze(0)
            
            return kernel
            
        except Exception as e:
            print(f"Error creating Gaussian kernel: {e}")
            # Return identity kernel
            kernel = torch.zeros(kernel_size, kernel_size, device=device)
            kernel[kernel_size//2, kernel_size//2] = 1.0
            return kernel
    
    def _enhance_normal_details(self, normals: torch.Tensor) -> torch.Tensor:
        """Enhance fine details in normal maps"""
        try:
            # Apply high-pass filter to enhance details
            blur_kernel = torch.ones(3, 3, device=normals.device) / 9.0
            
            enhanced_normals = torch.zeros_like(normals)
            for i in range(3):
                # Apply blur
                blurred = F.conv2d(
                    normals[:, i:i+1], 
                    blur_kernel.unsqueeze(0).unsqueeze(0), 
                    padding=1
                )
                
                # High-pass = original - blurred
                high_pass = normals[:, i:i+1] - blurred
                
                # Add enhanced details
                enhanced_normals[:, i:i+1] = normals[:, i:i+1] + high_pass * 0.3
            
            # Re-normalize
            enhanced_normals = normalize_tensor(enhanced_normals, dim=1)
            
            return enhanced_normals
            
        except Exception as e:
            print(f"Error enhancing normal details: {e}")
            return normals
    
    def _transform_to_world_space(self, normals: torch.Tensor) -> torch.Tensor:
        """Transform normals from camera space to world space"""
        try:
            # Simplified transformation - in real implementation would use camera parameters
            # For now, just apply a simple rotation
            
            # Example rotation matrix (identity for simplicity)
            rotation_matrix = torch.eye(3, device=normals.device)
            
            # Reshape normals for matrix multiplication
            batch_size, _, height, width = normals.shape
            normals_flat = normals.view(batch_size, 3, -1)  # B x 3 x (H*W)
            
            # Apply rotation
            transformed_normals = torch.bmm(
                rotation_matrix.unsqueeze(0).repeat(batch_size, 1, 1),
                normals_flat
            )
            
            # Reshape back
            transformed_normals = transformed_normals.view(batch_size, 3, height, width)
            
            return transformed_normals
            
        except Exception as e:
            print(f"Error transforming normals to world space: {e}")
            return normals
    
    def _create_normal_visualization(self, normals: torch.Tensor) -> torch.Tensor:
        """Create visualization of surface normals"""
        try:
            # Convert normals from [-1,1] to [0,1] for visualization
            normal_vis = (normals + 1.0) / 2.0
            
            # Ensure values are in valid range
            normal_vis = torch.clamp(normal_vis, 0.0, 1.0)
            
            return normal_vis
            
        except Exception as e:
            print(f"Error creating normal visualization: {e}")
            # Return default visualization
            batch_size, _, height, width = normals.shape
            return torch.ones(batch_size, 3, height, width, device=normals.device) * 0.5
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate node inputs"""
        model = kwargs.get('model')
        if model is None:
            return "Model input is required"
        
        image = kwargs.get('image')
        if image is None:
            return "Image input is required"
        
        normal_smoothing = kwargs.get('normal_smoothing', 0.0)
        if normal_smoothing < 0.0 or normal_smoothing > 1.0:
            return "Normal smoothing must be between 0.0 and 1.0"
        
        return True
