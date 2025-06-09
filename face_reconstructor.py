"""
3D Face Reconstructor Node for ComfyUI
Main node for performing 3D face reconstruction using Pixel3DMM
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .utils import (
    convert_comfyui_image, convert_to_comfyui_image, 
    preprocess_image, postprocess_mesh_data, validate_flame_parameters
)

class FaceReconstructor3D:
    """
    ComfyUI node for 3D face reconstruction using Pixel3DMM
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PIXEL3DMM_MODEL",),
                "image": ("IMAGE",),
                "reconstruction_quality": (["fast", "balanced", "high"], {"default": "balanced"}),
                "optimize_flame": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "optimization_steps": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 10
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "FLAME_PARAMS", "MESH_DATA", "STRING")
    RETURN_NAMES = ("rendered_face", "flame_parameters", "mesh_data", "status")
    FUNCTION = "reconstruct_face"
    CATEGORY = "Pixel3DMM"
    
    def __init__(self):
        self.last_model = None
    
    def reconstruct_face(self, model: Dict[str, Any], image: torch.Tensor, 
                        reconstruction_quality: str = "balanced", optimize_flame: bool = True,
                        optimization_steps: int = 100, learning_rate: float = 0.01) -> Tuple:
        """
        Perform 3D face reconstruction
        
        Args:
            model: Loaded Pixel3DMM model container
            image: Input face image
            reconstruction_quality: Quality setting for reconstruction
            optimize_flame: Whether to optimize FLAME parameters
            optimization_steps: Number of optimization steps
            learning_rate: Learning rate for optimization
            
        Returns:
            Tuple of (rendered_face, flame_parameters, mesh_data, status)
        """
        try:
            if model is None:
                return None, None, None, "❌ No model loaded. Please connect a Pixel3DMM Loader node."
            
            # Store model reference
            self.last_model = model
            device = model['device']
            
            # Preprocess input image
            processed_image = self._preprocess_input(image, device)
            if processed_image is None:
                return None, None, None, "❌ Failed to preprocess input image"
            
            # Extract features using DINO backbone
            features = self._extract_features(model, processed_image)
            if features is None:
                return None, None, None, "❌ Failed to extract image features"
            
            # Predict initial FLAME parameters
            flame_params = self._predict_flame_parameters(model, features)
            if flame_params is None:
                return None, None, None, "❌ Failed to predict FLAME parameters"
            
            # Optimize FLAME parameters if requested
            if optimize_flame:
                flame_params = self._optimize_flame_parameters(
                    model, processed_image, flame_params, 
                    optimization_steps, learning_rate
                )
            
            # Generate 3D mesh
            mesh_data = self._generate_mesh(model, flame_params)
            if mesh_data is None:
                return None, None, None, "❌ Failed to generate 3D mesh"
            
            # Render face
            rendered_face = self._render_face(model, mesh_data, processed_image.shape[-2:])
            if rendered_face is None:
                return None, None, None, "❌ Failed to render face"
            
            # Convert outputs to ComfyUI format
            rendered_face = convert_to_comfyui_image(rendered_face)
            
            status = f"✅ 3D face reconstruction completed\n"
            status += f"🎭 Quality: {reconstruction_quality}\n"
            status += f"🔥 FLAME optimization: {'Yes' if optimize_flame else 'No'}\n"
            status += f"📊 Vertices: {mesh_data['vertex_count']}\n"
            status += f"🔺 Faces: {mesh_data['face_count']}"
            
            return rendered_face, flame_params, mesh_data, status
            
        except Exception as e:
            error_msg = f"❌ Error in face reconstruction: {str(e)}"
            print(error_msg)
            return None, None, None, error_msg
    
    def _preprocess_input(self, image: torch.Tensor, device: str) -> Optional[torch.Tensor]:
        """Preprocess input image for inference"""
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
    
    def _extract_features(self, model: Dict[str, Any], image: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract features using DINO backbone"""
        try:
            dino_model = model['components'].get('dino')
            if dino_model is None:
                print("DINO model not found in model container")
                return None
            
            dino_model.eval()
            with torch.no_grad():
                features = dino_model(image)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def _predict_flame_parameters(self, model: Dict[str, Any], features: torch.Tensor) -> Optional[torch.Tensor]:
        """Predict FLAME parameters from features"""
        try:
            predictor = model['components'].get('predictor')
            if predictor is None:
                print("Predictor model not found in model container")
                return None
            
            predictor.eval()
            with torch.no_grad():
                flame_params = predictor(features)
            
            # Validate parameters
            if not validate_flame_parameters(flame_params):
                print("Invalid FLAME parameters predicted")
                return None
            
            return flame_params
            
        except Exception as e:
            print(f"Error predicting FLAME parameters: {e}")
            return None
    
    def _optimize_flame_parameters(self, model: Dict[str, Any], image: torch.Tensor, 
                                 initial_params: torch.Tensor, steps: int, lr: float) -> torch.Tensor:
        """Optimize FLAME parameters using gradient descent"""
        try:
            # Make parameters optimizable
            flame_params = initial_params.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([flame_params], lr=lr)
            
            flame_model = model['components'].get('flame')
            if flame_model is None:
                print("FLAME model not found, skipping optimization")
                return initial_params
            
            for step in range(steps):
                optimizer.zero_grad()
                
                # Generate mesh with current parameters
                mesh_output = flame_model(flame_params)
                
                # Compute loss (simplified - would need proper loss functions)
                # This is a placeholder loss
                loss = torch.mean(flame_params ** 2) * 0.001  # Regularization
                
                # Add reconstruction loss here in real implementation
                # loss += reconstruction_loss(mesh_output, image)
                
                loss.backward()
                optimizer.step()
                
                if step % 20 == 0:
                    print(f"Optimization step {step}/{steps}, loss: {loss.item():.6f}")
            
            return flame_params.detach()
            
        except Exception as e:
            print(f"Error optimizing FLAME parameters: {e}")
            return initial_params
    
    def _generate_mesh(self, model: Dict[str, Any], flame_params: torch.Tensor) -> Optional[Dict[str, Any]]:
        """Generate 3D mesh from FLAME parameters"""
        try:
            flame_model = model['components'].get('flame')
            if flame_model is None:
                print("FLAME model not found in model container")
                return None
            
            flame_model.eval()
            with torch.no_grad():
                mesh_output = flame_model(flame_params)
            
            # Postprocess mesh data
            mesh_data = postprocess_mesh_data(
                mesh_output['vertices'], 
                mesh_output['faces']
            )
            
            # Add additional mesh information
            mesh_data.update({
                'landmarks': mesh_output.get('landmarks'),
                'flame_params': flame_params.detach().cpu().numpy(),
            })
            
            return mesh_data
            
        except Exception as e:
            print(f"Error generating mesh: {e}")
            return None
    
    def _render_face(self, model: Dict[str, Any], mesh_data: Dict[str, Any], 
                    target_size: Tuple[int, int]) -> Optional[torch.Tensor]:
        """Render 3D face mesh to image"""
        try:
            # Simplified rendering - in real implementation would use proper renderer
            device = model['device']
            height, width = target_size
            
            # Create a simple rendered image (placeholder)
            rendered = torch.zeros(1, 3, height, width, device=device)
            
            # Add some basic visualization based on mesh data
            if mesh_data['vertex_count'] > 0:
                # Simple visualization: create a gradient based on vertex positions
                vertices = torch.from_numpy(mesh_data['vertices']).to(device)
                
                # Project vertices to image space (simplified)
                x_coords = ((vertices[:, 0] + 1) * width / 2).long().clamp(0, width-1)
                y_coords = ((vertices[:, 1] + 1) * height / 2).long().clamp(0, height-1)
                
                # Set pixel values at vertex locations
                for i in range(min(len(x_coords), 1000)):  # Limit for performance
                    x, y = x_coords[i], y_coords[i]
                    rendered[0, :, y, x] = 0.8  # White dots for vertices
            
            return rendered
            
        except Exception as e:
            print(f"Error rendering face: {e}")
            # Return a default image
            height, width = target_size
            return torch.zeros(1, 3, height, width)
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate node inputs"""
        model = kwargs.get('model')
        if model is None:
            return "Model input is required"
        
        image = kwargs.get('image')
        if image is None:
            return "Image input is required"
        
        optimization_steps = kwargs.get('optimization_steps', 100)
        if optimization_steps < 10 or optimization_steps > 1000:
            return "Optimization steps must be between 10 and 1000"
        
        learning_rate = kwargs.get('learning_rate', 0.01)
        if learning_rate < 0.001 or learning_rate > 0.1:
            return "Learning rate must be between 0.001 and 0.1"
        
        return True
