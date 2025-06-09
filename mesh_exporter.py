"""
3D Mesh Exporter Node for ComfyUI
Exports 3D mesh data to various formats
"""

import torch
import numpy as np
import os
from typing import Dict, Any, Tuple, Optional
from .utils import postprocess_mesh_data

class MeshExporter:
    """
    ComfyUI node for exporting 3D mesh data to files
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("MESH_DATA",),
                "output_format": (["obj", "ply", "stl"], {"default": "obj"}),
                "filename": ("STRING", {
                    "default": "face_mesh",
                    "multiline": False
                }),
            },
            "optional": {
                "output_directory": ("STRING", {
                    "default": "output/meshes",
                    "multiline": False
                }),
                "include_textures": ("BOOLEAN", {"default": False}),
                "scale_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
                "center_mesh": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "status")
    FUNCTION = "export_mesh"
    CATEGORY = "Pixel3DMM"
    
    def __init__(self):
        self.supported_formats = ['obj', 'ply', 'stl']
    
    def export_mesh(self, mesh_data: Dict[str, Any], output_format: str = "obj",
                   filename: str = "face_mesh", output_directory: str = "output/meshes",
                   include_textures: bool = False, scale_factor: float = 1.0,
                   center_mesh: bool = True) -> Tuple[str, str]:
        """
        Export 3D mesh to file
        
        Args:
            mesh_data: Mesh data containing vertices and faces
            output_format: Output file format (obj/ply/stl)
            filename: Output filename (without extension)
            output_directory: Output directory path
            include_textures: Whether to include texture coordinates
            scale_factor: Scale factor for mesh
            center_mesh: Whether to center the mesh at origin
            
        Returns:
            Tuple of (file_path, status_message)
        """
        try:
            if mesh_data is None:
                return "", "❌ No mesh data provided"
            
            if output_format not in self.supported_formats:
                return "", f"❌ Unsupported format: {output_format}"
            
            # Ensure output directory exists
            os.makedirs(output_directory, exist_ok=True)
            
            # Process mesh data
            processed_mesh = self._process_mesh_data(mesh_data, scale_factor, center_mesh)
            if processed_mesh is None:
                return "", "❌ Failed to process mesh data"
            
            # Generate output file path
            file_path = os.path.join(output_directory, f"{filename}.{output_format}")
            
            # Export based on format
            success = False
            if output_format == "obj":
                success = self._export_obj(processed_mesh, file_path, include_textures)
            elif output_format == "ply":
                success = self._export_ply(processed_mesh, file_path)
            elif output_format == "stl":
                success = self._export_stl(processed_mesh, file_path)
            
            if not success:
                return "", f"❌ Failed to export mesh as {output_format.upper()}"
            
            # Generate status message
            status = f"✅ Mesh exported successfully\n"
            status += f"📁 File: {file_path}\n"
            status += f"📊 Format: {output_format.upper()}\n"
            status += f"🔺 Vertices: {processed_mesh['vertex_count']}\n"
            status += f"🔺 Faces: {processed_mesh['face_count']}\n"
            status += f"📏 Scale: {scale_factor}x\n"
            status += f"🎯 Centered: {'Yes' if center_mesh else 'No'}"
            
            return file_path, status
            
        except Exception as e:
            error_msg = f"❌ Error exporting mesh: {str(e)}"
            print(error_msg)
            return "", error_msg
    
    def _process_mesh_data(self, mesh_data: Dict[str, Any], scale_factor: float,
                          center_mesh: bool) -> Optional[Dict[str, Any]]:
        """Process mesh data before export"""
        try:
            # Extract vertices and faces
            vertices = mesh_data.get('vertices')
            faces = mesh_data.get('faces')
            
            if vertices is None or faces is None:
                print("Missing vertices or faces in mesh data")
                return None
            
            # Convert to numpy if needed
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.detach().cpu().numpy()
            if isinstance(faces, torch.Tensor):
                faces = faces.detach().cpu().numpy()
            
            # Handle batch dimension
            if vertices.ndim == 3:
                vertices = vertices[0]  # Take first batch
            if faces.ndim == 3:
                faces = faces[0]
            
            # Apply scale factor
            if scale_factor != 1.0:
                vertices = vertices * scale_factor
            
            # Center mesh if requested
            if center_mesh:
                centroid = np.mean(vertices, axis=0)
                vertices = vertices - centroid
            
            # Prepare processed mesh data
            processed_mesh = {
                'vertices': vertices,
                'faces': faces,
                'vertex_count': vertices.shape[0],
                'face_count': faces.shape[0],
            }
            
            # Include additional data if available
            if 'landmarks' in mesh_data:
                landmarks = mesh_data['landmarks']
                if isinstance(landmarks, torch.Tensor):
                    landmarks = landmarks.detach().cpu().numpy()
                if landmarks.ndim == 3:
                    landmarks = landmarks[0]
                
                if center_mesh:
                    landmarks = landmarks - centroid
                if scale_factor != 1.0:
                    landmarks = landmarks * scale_factor
                
                processed_mesh['landmarks'] = landmarks
            
            if 'flame_params' in mesh_data:
                processed_mesh['flame_params'] = mesh_data['flame_params']
            
            return processed_mesh
            
        except Exception as e:
            print(f"Error processing mesh data: {e}")
            return None
    
    def _export_obj(self, mesh_data: Dict[str, Any], file_path: str, 
                   include_textures: bool) -> bool:
        """Export mesh as OBJ format"""
        try:
            vertices = mesh_data['vertices']
            faces = mesh_data['faces']
            
            with open(file_path, 'w') as f:
                # Write header
                f.write("# OBJ file generated by Pixel3DMM ComfyUI node\n")
                f.write(f"# Vertices: {mesh_data['vertex_count']}\n")
                f.write(f"# Faces: {mesh_data['face_count']}\n\n")
                
                # Write vertices
                for vertex in vertices:
                    f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                
                # Write texture coordinates if available and requested
                if include_textures and 'uv_coords' in mesh_data:
                    uv_coords = mesh_data['uv_coords']
                    for uv in uv_coords:
                        f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
                
                # Write faces (OBJ uses 1-based indexing)
                f.write("\n")
                for face in faces:
                    if include_textures and 'uv_coords' in mesh_data:
                        f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")
                    else:
                        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                
                # Write landmarks as comments if available
                if 'landmarks' in mesh_data:
                    f.write("\n# Facial landmarks\n")
                    landmarks = mesh_data['landmarks']
                    for i, landmark in enumerate(landmarks):
                        f.write(f"# landmark_{i}: {landmark[0]:.6f} {landmark[1]:.6f} {landmark[2]:.6f}\n")
            
            return True
            
        except Exception as e:
            print(f"Error exporting OBJ: {e}")
            return False
    
    def _export_ply(self, mesh_data: Dict[str, Any], file_path: str) -> bool:
        """Export mesh as PLY format"""
        try:
            vertices = mesh_data['vertices']
            faces = mesh_data['faces']
            vertex_count = mesh_data['vertex_count']
            face_count = mesh_data['face_count']
            
            with open(file_path, 'w') as f:
                # Write PLY header
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"comment Generated by Pixel3DMM ComfyUI node\n")
                f.write(f"element vertex {vertex_count}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write(f"element face {face_count}\n")
                f.write("property list uchar int vertex_indices\n")
                f.write("end_header\n")
                
                # Write vertices
                for vertex in vertices:
                    f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                
                # Write faces
                for face in faces:
                    f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
            
            return True
            
        except Exception as e:
            print(f"Error exporting PLY: {e}")
            return False
    
    def _export_stl(self, mesh_data: Dict[str, Any], file_path: str) -> bool:
        """Export mesh as STL format"""
        try:
            vertices = mesh_data['vertices']
            faces = mesh_data['faces']
            
            with open(file_path, 'w') as f:
                # Write STL header
                f.write("solid Pixel3DMM_Mesh\n")
                
                # Write triangles
                for face in faces:
                    # Get face vertices
                    v1 = vertices[face[0]]
                    v2 = vertices[face[1]]
                    v3 = vertices[face[2]]
                    
                    # Compute face normal
                    edge1 = v2 - v1
                    edge2 = v3 - v1
                    normal = np.cross(edge1, edge2)
                    normal = normal / (np.linalg.norm(normal) + 1e-8)
                    
                    # Write facet
                    f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                    f.write("    outer loop\n")
                    f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
                    f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
                    f.write(f"      vertex {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n")
                    f.write("    endloop\n")
                    f.write("  endfacet\n")
                
                f.write("endsolid Pixel3DMM_Mesh\n")
            
            return True
            
        except Exception as e:
            print(f"Error exporting STL: {e}")
            return False
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate node inputs"""
        mesh_data = kwargs.get('mesh_data')
        if mesh_data is None:
            return "Mesh data input is required"
        
        output_format = kwargs.get('output_format', 'obj')
        if output_format not in ['obj', 'ply', 'stl']:
            return "Invalid output format"
        
        filename = kwargs.get('filename', '')
        if not filename or not filename.strip():
            return "Filename cannot be empty"
        
        scale_factor = kwargs.get('scale_factor', 1.0)
        if scale_factor <= 0:
            return "Scale factor must be positive"
        
        return True
