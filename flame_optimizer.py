"""
FLAME Parameter Optimizer Node for ComfyUI
Optimizes FLAME parameters using UV and normal constraints
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .utils import validate_flame_parameters

class FLAMEOptimizer:
    """
    ComfyUI node for optimizing FLAME parameters using geometric constraints
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PIXEL3DMM_MODEL",),
                "image": ("IMAGE",),
                "initial_params": ("FLAME_PARAMS",),
                "optimization_steps": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 10
                }),
            },
            "optional": {
                "uv_coordinates": ("UV_COORDS",),
                "surface_normals": ("NORMALS",),
                "learning_rate": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001
                }),
                "uv_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "normal_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "regularization_weight": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001
                }),
            }
        }
    
    RETURN_TYPES = ("FLAME_PARAMS", "MESH_DATA", "STRING")
    RETURN_NAMES = ("optimized_params", "mesh_data", "status")
    FUNCTION = "optimize_flame"
    CATEGORY = "Pixel3DMM"
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_flame(self, model: Dict[str, Any], image: torch.Tensor, 
                      initial_params: torch.Tensor, optimization_steps: int = 100,
                      uv_coordinates: Optional[torch.Tensor] = None,
                      surface_normals: Optional[torch.Tensor] = None,
                      learning_rate: float = 0.01, uv_weight: float = 1.0,
                      normal_weight: float = 1.0, regularization_weight: float = 0.01) -> Tuple:
        """
        Optimize FLAME parameters using geometric constraints
        
        Args:
            model: Loaded Pixel3DMM model container
            image: Input face image
            initial_params: Initial FLAME parameters
            optimization_steps: Number of optimization steps
            uv_coordinates: Optional UV coordinate constraints
            surface_normals: Optional surface normal constraints
            learning_rate: Learning rate for optimization
            uv_weight: Weight for UV loss term
            normal_weight: Weight for normal loss term
            regularization_weight: Weight for regularization term
            
        Returns:
            Tuple of (optimized_params, mesh_data, status)
        """
        try:
            if model is None:
                return None, None, "❌ No model loaded. Please connect a Pixel3DMM Loader node."
            
            if not validate_flame_parameters(initial_params):
                return None, None, "❌ Invalid initial FLAME parameters"
            
            device = model['device']
            
            # Move inputs to device
            initial_params = initial_params.to(device)
            if uv_coordinates is not None:
                uv_coordinates = uv_coordinates.to(device)
            if surface_normals is not None:
                surface_normals = surface_normals.to(device)
            
            # Initialize optimization
            optimized_params = self._initialize_optimization(initial_params)
            
            # Get FLAME model
            flame_model = model['components'].get('flame')
            if flame_model is None:
                return None, None, "❌ FLAME model not found in model container"
            
            # Run optimization
            optimized_params, optimization_info = self._run_optimization(
                flame_model, optimized_params, uv_coordinates, surface_normals,
                optimization_steps, learning_rate, uv_weight, normal_weight, regularization_weight
            )
            
            # Generate final mesh
            mesh_data = self._generate_final_mesh(flame_model, optimized_params)
            
            status = f"✅ FLAME optimization completed\n"
            status += f"🔄 Steps: {optimization_steps}\n"
            status += f"📈 Learning rate: {learning_rate}\n"
            status += f"📊 Final loss: {optimization_info['final_loss']:.6f}\n"
            status += f"🎯 UV weight: {uv_weight}, Normal weight: {normal_weight}\n"
            status += f"⚖️ Regularization: {regularization_weight}"
            
            return optimized_params, mesh_data, status
            
        except Exception as e:
            error_msg = f"❌ Error in FLAME optimization: {str(e)}"
            print(error_msg)
            return None, None, error_msg
    
    def _initialize_optimization(self, initial_params: torch.Tensor) -> torch.Tensor:
        """Initialize parameters for optimization"""
        try:
            # Clone parameters and enable gradients
            optimized_params = initial_params.clone().detach().requires_grad_(True)
            
            return optimized_params
            
        except Exception as e:
            print(f"Error initializing optimization: {e}")
            return initial_params
    
    def _run_optimization(self, flame_model: torch.nn.Module, params: torch.Tensor,
                         uv_coords: Optional[torch.Tensor], normals: Optional[torch.Tensor],
                         steps: int, lr: float, uv_weight: float, normal_weight: float,
                         reg_weight: float) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run the optimization loop"""
        try:
            # Initialize optimizer
            optimizer = torch.optim.Adam([params], lr=lr)
            
            # Track optimization progress
            losses = []
            
            for step in range(steps):
                optimizer.zero_grad()
                
                # Forward pass through FLAME
                mesh_output = flame_model(params)
                
                # Compute losses
                total_loss = 0.0
                loss_components = {}
                
                # UV loss
                if uv_coords is not None and uv_weight > 0:
                    uv_loss = self._compute_uv_loss(mesh_output, uv_coords)
                    total_loss += uv_weight * uv_loss
                    loss_components['uv_loss'] = uv_loss.item()
                
                # Normal loss
                if normals is not None and normal_weight > 0:
                    normal_loss = self._compute_normal_loss(mesh_output, normals)
                    total_loss += normal_weight * normal_loss
                    loss_components['normal_loss'] = normal_loss.item()
                
                # Regularization loss
                if reg_weight > 0:
                    reg_loss = self._compute_regularization_loss(params)
                    total_loss += reg_weight * reg_loss
                    loss_components['reg_loss'] = reg_loss.item()
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Track progress
                loss_components['total_loss'] = total_loss.item()
                losses.append(loss_components)
                
                # Print progress
                if step % 20 == 0:
                    print(f"Step {step}/{steps}: Loss = {total_loss.item():.6f}")
            
            optimization_info = {
                'final_loss': losses[-1]['total_loss'] if losses else 0.0,
                'loss_history': losses
            }
            
            return params.detach(), optimization_info
            
        except Exception as e:
            print(f"Error in optimization loop: {e}")
            return params.detach(), {'final_loss': float('inf'), 'loss_history': []}
    
    def _compute_uv_loss(self, mesh_output: Dict[str, torch.Tensor], 
                        target_uv: torch.Tensor) -> torch.Tensor:
        """Compute UV coordinate loss"""
        try:
            # Get vertices from mesh output
            vertices = mesh_output['vertices']  # B x N x 3
            
            # Project vertices to image space (simplified)
            # In real implementation, would use proper camera projection
            projected_vertices = vertices[:, :, :2]  # Use X,Y coordinates
            
            # Convert to UV space [0,1]
            projected_uv = (projected_vertices + 1.0) / 2.0
            
            # Sample target UV at vertex locations (simplified)
            # This is a placeholder - real implementation would use proper UV sampling
            batch_size = target_uv.shape[0]
            height, width = target_uv.shape[-2:]
            
            # Create sampling grid
            grid_x = torch.linspace(-1, 1, width, device=target_uv.device)
            grid_y = torch.linspace(-1, 1, height, device=target_uv.device)
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
            
            # Sample UV coordinates
            sampled_uv = F.grid_sample(target_uv, grid, align_corners=False)
            
            # Compute loss (simplified)
            uv_loss = F.mse_loss(projected_uv.mean(dim=1), sampled_uv.mean(dim=[2, 3]))
            
            return uv_loss
            
        except Exception as e:
            print(f"Error computing UV loss: {e}")
            return torch.tensor(0.0, device=mesh_output['vertices'].device)
    
    def _compute_normal_loss(self, mesh_output: Dict[str, torch.Tensor], 
                           target_normals: torch.Tensor) -> torch.Tensor:
        """Compute surface normal loss"""
        try:
            # Get vertices and faces
            vertices = mesh_output['vertices']  # B x N x 3
            faces = mesh_output['faces']        # F x 3
            
            # Compute face normals (simplified)
            face_normals = self._compute_face_normals(vertices, faces)
            
            # Project normals to image space and compare with target
            # This is a simplified implementation
            
            # Average face normals as a simple loss
            predicted_normal = face_normals.mean(dim=1)  # B x 3
            target_normal = target_normals.mean(dim=[2, 3])  # B x 3
            
            normal_loss = F.mse_loss(predicted_normal, target_normal)
            
            return normal_loss
            
        except Exception as e:
            print(f"Error computing normal loss: {e}")
            return torch.tensor(0.0, device=mesh_output['vertices'].device)
    
    def _compute_face_normals(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """Compute face normals from vertices and faces"""
        try:
            batch_size = vertices.shape[0]
            num_faces = faces.shape[0]
            
            # Get face vertices
            face_vertices = vertices[:, faces]  # B x F x 3 x 3
            
            # Compute edge vectors
            v1 = face_vertices[:, :, 1] - face_vertices[:, :, 0]  # B x F x 3
            v2 = face_vertices[:, :, 2] - face_vertices[:, :, 0]  # B x F x 3
            
            # Compute cross product for normals
            normals = torch.cross(v1, v2, dim=-1)  # B x F x 3
            
            # Normalize
            normals = F.normalize(normals, p=2.0, dim=-1)
            
            return normals
            
        except Exception as e:
            print(f"Error computing face normals: {e}")
            batch_size = vertices.shape[0]
            return torch.zeros(batch_size, faces.shape[0], 3, device=vertices.device)
    
    def _compute_regularization_loss(self, params: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss to keep parameters reasonable"""
        try:
            # L2 regularization on parameters
            reg_loss = torch.mean(params ** 2)
            
            return reg_loss
            
        except Exception as e:
            print(f"Error computing regularization loss: {e}")
            return torch.tensor(0.0, device=params.device)
    
    def _generate_final_mesh(self, flame_model: torch.nn.Module, 
                           params: torch.Tensor) -> Optional[Dict[str, Any]]:
        """Generate final mesh from optimized parameters"""
        try:
            flame_model.eval()
            with torch.no_grad():
                mesh_output = flame_model(params)
            
            # Convert to numpy for export
            mesh_data = {
                'vertices': mesh_output['vertices'].cpu().numpy(),
                'faces': mesh_output['faces'].cpu().numpy(),
                'vertex_count': mesh_output['vertices'].shape[1],
                'face_count': mesh_output['faces'].shape[0],
                'flame_params': params.cpu().numpy(),
            }
            
            if 'landmarks' in mesh_output:
                mesh_data['landmarks'] = mesh_output['landmarks'].cpu().numpy()
            
            return mesh_data
            
        except Exception as e:
            print(f"Error generating final mesh: {e}")
            return None
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate node inputs"""
        model = kwargs.get('model')
        if model is None:
            return "Model input is required"
        
        image = kwargs.get('image')
        if image is None:
            return "Image input is required"
        
        initial_params = kwargs.get('initial_params')
        if initial_params is None:
            return "Initial FLAME parameters are required"
        
        optimization_steps = kwargs.get('optimization_steps', 100)
        if optimization_steps < 10 or optimization_steps > 1000:
            return "Optimization steps must be between 10 and 1000"
        
        learning_rate = kwargs.get('learning_rate', 0.01)
        if learning_rate < 0.001 or learning_rate > 0.1:
            return "Learning rate must be between 0.001 and 0.1"
        
        return True
