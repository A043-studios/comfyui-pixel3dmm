"""
Utility functions for Pixel3DMM ComfyUI nodes
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import os
from typing import Optional, Tuple, Union, Any

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image"""
    try:
        if isinstance(tensor, torch.Tensor):
            # Handle different tensor formats
            if tensor.dim() == 4:  # BCHW
                tensor = tensor.squeeze(0)
            if tensor.dim() == 3:  # CHW
                tensor = tensor.permute(1, 2, 0)  # HWC
            
            # Ensure values are in [0, 1] range
            tensor = torch.clamp(tensor, 0, 1)
            
            # Convert to numpy
            if tensor.requires_grad:
                tensor = tensor.detach()
            tensor = tensor.cpu().numpy()
            
            # Convert to uint8
            tensor = (tensor * 255).astype(np.uint8)
            
            # Handle grayscale vs RGB
            if tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)
                return Image.fromarray(tensor, mode='L')
            elif tensor.shape[-1] == 3:
                return Image.fromarray(tensor, mode='RGB')
            elif tensor.shape[-1] == 4:
                return Image.fromarray(tensor, mode='RGBA')
            else:
                # Convert single channel to RGB
                if len(tensor.shape) == 2:
                    return Image.fromarray(tensor, mode='L')
                
        return Image.fromarray(tensor)
        
    except Exception as e:
        print(f"Error converting tensor to PIL: {e}")
        # Return a default image on error
        return Image.new('RGB', (512, 512), color='black')

def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor and rearrange dimensions
        tensor = torch.from_numpy(image_np)
        
        # Ensure HWC format, then convert to CHW
        if tensor.dim() == 3:
            tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)  # CHW -> BCHW
        
        return tensor
        
    except Exception as e:
        print(f"Error converting PIL to tensor: {e}")
        # Return default tensor on error
        return torch.zeros(1, 3, 512, 512)

def preprocess_image(image: torch.Tensor, target_size: int = 512) -> torch.Tensor:
    """Preprocess image for Pixel3DMM inference"""
    try:
        # Ensure tensor is in BCHW format
        if image.dim() == 3:  # CHW
            image = image.unsqueeze(0)  # Add batch dimension
        elif image.dim() == 2:  # HW
            image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Resize to target size
        if image.shape[-2:] != (target_size, target_size):
            image = F.interpolate(
                image, 
                size=(target_size, target_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Ensure values are in [0, 1] range
        image = torch.clamp(image, 0, 1)
        
        # Ensure 3 channels (RGB)
        if image.shape[1] == 1:  # Grayscale
            image = image.repeat(1, 3, 1, 1)
        elif image.shape[1] == 4:  # RGBA
            image = image[:, :3, :, :]  # Remove alpha channel
        
        return image
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        # Return default tensor on error
        return torch.zeros(1, 3, target_size, target_size)

def postprocess_mesh_data(vertices: torch.Tensor, faces: torch.Tensor) -> dict:
    """Postprocess mesh data for export"""
    try:
        # Convert to numpy if needed
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.detach().cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.detach().cpu().numpy()
        
        # Ensure correct shapes
        if vertices.ndim == 3:  # Batch dimension
            vertices = vertices[0]
        if faces.ndim == 3:  # Batch dimension
            faces = faces[0]
        
        return {
            'vertices': vertices,
            'faces': faces,
            'vertex_count': vertices.shape[0],
            'face_count': faces.shape[0]
        }
        
    except Exception as e:
        print(f"Error postprocessing mesh data: {e}")
        return {
            'vertices': np.zeros((0, 3)),
            'faces': np.zeros((0, 3)),
            'vertex_count': 0,
            'face_count': 0
        }

def normalize_tensor(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize tensor along specified dimension"""
    try:
        return F.normalize(tensor, p=2.0, dim=dim)
    except Exception as e:
        print(f"Error normalizing tensor: {e}")
        return tensor

def safe_load_model(model_path: str, device: str = 'cpu') -> Optional[torch.nn.Module]:
    """Safely load a PyTorch model"""
    try:
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None
        
        # Load model with appropriate device mapping
        model = torch.load(model_path, map_location=device)
        
        # Handle different model formats
        if isinstance(model, dict):
            # If it's a state dict, we need the model architecture
            print("Warning: Loaded state dict, need model architecture")
            return None
        
        # Set to evaluation mode
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def create_default_config() -> dict:
    """Create default configuration for Pixel3DMM"""
    return {
        'model': {
            'encoder_backbone': 'vit_base_patch14_dinov2.lvd142m',
            'finetune_backbone': False,
            'embedding_dim': 128,
            'flame_dim': 101,
        },
        'image_size': [512, 512],
        'device': 'cpu',
        'batch_size': 1,
    }

def validate_flame_parameters(params: torch.Tensor) -> bool:
    """Validate FLAME parameter tensor"""
    try:
        if not isinstance(params, torch.Tensor):
            return False
        
        # Check shape - should be [batch_size, 101] for full FLAME params
        if params.dim() != 2 or params.shape[-1] != 101:
            return False
        
        # Check for NaN or infinite values
        if torch.isnan(params).any() or torch.isinf(params).any():
            return False
        
        return True
        
    except Exception:
        return False

def convert_comfyui_image(image: Any) -> torch.Tensor:
    """Convert ComfyUI image format to standard tensor"""
    try:
        if isinstance(image, torch.Tensor):
            # ComfyUI typically uses BHWC format
            if image.dim() == 4:  # BHWC
                return image.permute(0, 3, 1, 2)  # BHWC -> BCHW
            elif image.dim() == 3:  # HWC
                return image.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
        
        # Handle other formats
        if isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image.astype(np.float32))
            if tensor.dim() == 3:  # HWC
                return tensor.permute(2, 0, 1).unsqueeze(0)
        
        if isinstance(image, Image.Image):
            return pil_to_tensor(image)
        
        # Default fallback
        return torch.zeros(1, 3, 512, 512)
        
    except Exception as e:
        print(f"Error converting ComfyUI image: {e}")
        return torch.zeros(1, 3, 512, 512)

def convert_to_comfyui_image(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor to ComfyUI image format (BHWC)"""
    try:
        if tensor.dim() == 4:  # BCHW
            return tensor.permute(0, 2, 3, 1)  # BCHW -> BHWC
        elif tensor.dim() == 3:  # CHW
            return tensor.permute(1, 2, 0).unsqueeze(0)  # CHW -> BHWC
        
        return tensor
        
    except Exception as e:
        print(f"Error converting to ComfyUI format: {e}")
        return torch.zeros(1, 512, 512, 3)

def get_device_info() -> dict:
    """Get device information for model loading"""
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    if torch.cuda.is_available():
        device_info['gpu_name'] = torch.cuda.get_device_name(0)
        device_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
    
    return device_info
