# Installation Guide - Pixel3DMM ComfyUI Nodes

## 🚀 Quick Installation

### Method 1: ComfyUI Manager (Recommended)

1. **Open ComfyUI Manager** in your ComfyUI interface
2. **Search** for "Pixel3DMM" in the custom nodes section
3. **Click Install** and wait for completion
4. **Restart ComfyUI**

### Method 2: Git Clone (Manual)

```bash
# Navigate to ComfyUI custom nodes directory
cd /path/to/ComfyUI/custom_nodes/

# Clone the repository
git clone https://github.com/your-repo/comfyui-pixel3dmm.git

# Install dependencies
cd comfyui-pixel3dmm
pip install -r requirements.txt

# Restart ComfyUI
```

### Method 3: Download and Extract

1. **Download** the latest release from [GitHub Releases](https://github.com/your-repo/comfyui-pixel3dmm/releases)
2. **Extract** the zip file to `ComfyUI/custom_nodes/comfyui-pixel3dmm/`
3. **Install dependencies**:
   ```bash
   cd ComfyUI/custom_nodes/comfyui-pixel3dmm/
   pip install -r requirements.txt
   ```
4. **Restart ComfyUI**

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux
- **Python**: 3.9 or higher
- **RAM**: 8GB system memory
- **Storage**: 2GB free space
- **ComfyUI**: Latest version

### Recommended Requirements
- **OS**: Windows 11, macOS 12+, or Ubuntu 20.04+
- **Python**: 3.10 or higher
- **RAM**: 16GB+ system memory
- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better)
- **Storage**: 10GB free space (for models and outputs)
- **ComfyUI**: Latest stable version

## 🔧 Dependency Installation

### Core Dependencies

The following packages will be installed automatically:

```bash
# Core ML frameworks
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0

# Image processing
pillow>=8.3.0
opencv-python>=4.5.0
scikit-image>=0.18.0

# 3D processing
trimesh>=3.15.0

# Utilities
omegaconf>=2.1.0
mediapy>=1.1.0
```

### Optional Dependencies

For advanced features, you may want to install:

```bash
# Advanced vision transformers
pip install timm>=0.6.0

# Advanced 3D operations
pip install pytorch3d>=0.7.0

# 3D visualization
pip install pyvista>=0.37.0

# Camera operations
pip install dreifus>=0.1.0
```

## 🎯 Model Files

### Required Models

Download the following model files and place them in your models directory:

1. **Pixel3DMM Base Model** (Required)
   - File: `pixel3dmm_model.pth`
   - Size: ~500MB
   - Download: [Model Repository](https://github.com/your-repo/pixel3dmm-models)
   - Path: `ComfyUI/models/pixel3dmm/pixel3dmm_model.pth`

2. **FLAME Model Files** (Required)
   - Files: `FLAME2020/generic_model.pkl`, `FLAME_masks.pkl`
   - Size: ~50MB
   - Download: [FLAME Official](https://flame.is.tue.mpg.de/)
   - Path: `ComfyUI/models/flame/`

3. **DINO Backbone** (Optional - auto-downloaded)
   - Model: `vit_base_patch14_dinov2.lvd142m`
   - Size: ~350MB
   - Auto-downloaded on first use

### Model Directory Structure

```
ComfyUI/
├── models/
│   ├── pixel3dmm/
│   │   ├── pixel3dmm_model.pth
│   │   ├── uv_predictor.pth (optional)
│   │   └── normal_predictor.pth (optional)
│   ├── flame/
│   │   ├── FLAME2020/
│   │   │   └── generic_model.pkl
│   │   └── FLAME_masks.pkl
│   └── dino/
│       └── (auto-downloaded)
└── custom_nodes/
    └── comfyui-pixel3dmm/
        └── (this package)
```

## 🔍 Verification

### Test Installation

1. **Start ComfyUI** and check the console for any error messages
2. **Look for Pixel3DMM nodes** in the node browser under "Pixel3DMM" category
3. **Create a simple workflow**:
   ```
   Load Image → Pixel3DMM Loader → Face Reconstructor 3D
   ```
4. **Run the workflow** with a test image

### Expected Nodes

You should see these nodes in ComfyUI:

- 🔧 **Pixel3DMM Model Loader**
- 🎭 **3D Face Reconstructor**
- 🗺️ **UV Coordinate Predictor**
- 📐 **Surface Normal Predictor**
- 🔥 **FLAME Parameter Optimizer**
- 📦 **3D Mesh Exporter**

## 🛠️ Troubleshooting Installation

### Common Issues

#### ❌ "No module named 'pixel3dmm_loader'"

**Cause**: Installation incomplete or path issues

**Solutions**:
1. Ensure the package is in the correct directory
2. Restart ComfyUI completely
3. Check file permissions
4. Reinstall dependencies: `pip install -r requirements.txt`

#### ❌ "CUDA not available" or GPU errors

**Cause**: PyTorch not installed with CUDA support

**Solutions**:
1. Install PyTorch with CUDA:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
2. Or use CPU-only version:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

#### ❌ "Model file not found"

**Cause**: Missing model files

**Solutions**:
1. Download required model files (see Model Files section)
2. Check model paths in node configuration
3. Ensure proper directory structure

#### ❌ "Permission denied" errors

**Cause**: File permission issues

**Solutions**:
1. Run with administrator/sudo privileges
2. Check file ownership and permissions
3. Use virtual environment

### Getting Help

If you encounter issues:

1. **Check ComfyUI console** for detailed error messages
2. **Verify system requirements** are met
3. **Update ComfyUI** to the latest version
4. **Create an issue** on [GitHub](https://github.com/your-repo/comfyui-pixel3dmm/issues)
5. **Join our Discord** for community support

## 🔄 Updating

### Update via ComfyUI Manager

1. Open ComfyUI Manager
2. Go to "Update" tab
3. Find "Pixel3DMM" and click "Update"
4. Restart ComfyUI

### Manual Update

```bash
cd ComfyUI/custom_nodes/comfyui-pixel3dmm/
git pull origin main
pip install -r requirements.txt --upgrade
```

## 🗑️ Uninstallation

### Remove via ComfyUI Manager

1. Open ComfyUI Manager
2. Find "Pixel3DMM" in installed nodes
3. Click "Uninstall"
4. Restart ComfyUI

### Manual Removal

```bash
# Remove the package directory
rm -rf ComfyUI/custom_nodes/comfyui-pixel3dmm/

# Optional: Remove model files
rm -rf ComfyUI/models/pixel3dmm/
rm -rf ComfyUI/models/flame/
```

## 📞 Support

- **Documentation**: [Full documentation](https://github.com/your-repo/comfyui-pixel3dmm/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-repo/comfyui-pixel3dmm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/comfyui-pixel3dmm/discussions)
- **Discord**: [Community Discord](https://discord.gg/xxxxx)

---

**Installation complete!** 🎉 You're ready to create amazing 3D face reconstructions with ComfyUI!
