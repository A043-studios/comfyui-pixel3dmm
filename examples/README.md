# Example Workflows - Pixel3DMM ComfyUI Nodes

This directory contains example ComfyUI workflows demonstrating various use cases for the Pixel3DMM nodes.

## 📁 Available Examples

### 1. **basic_reconstruction.json**
**Purpose**: Simple 3D face reconstruction from a single image

**Workflow**:
```
Load Image → Pixel3DMM Loader → Face Reconstructor 3D → Preview + Export
```

**Features**:
- Quick setup for beginners
- Balanced quality settings
- Basic mesh export to OBJ format
- Estimated time: 30-60 seconds

**Use Cases**:
- Quick prototyping
- Learning the basic workflow
- Simple 3D avatar creation

### 2. **advanced_optimization.json**
**Purpose**: High-quality reconstruction with multi-constraint optimization

**Workflow**:
```
Load Image → Pixel3DMM Loader → Face Reconstructor 3D
                ↓
UV Predictor + Normal Predictor → FLAME Optimizer → Export
```

**Features**:
- High-quality reconstruction
- UV and normal constraint optimization
- Professional mesh export with textures
- Estimated time: 2-5 minutes

**Use Cases**:
- Professional 3D modeling
- High-quality avatar creation
- Research and development

## 🚀 How to Use Examples

### Method 1: Direct Import

1. **Open ComfyUI**
2. **Drag and drop** the JSON file into the ComfyUI interface
3. **Configure paths** (model files, input images, output directories)
4. **Queue the workflow**

### Method 2: Load via Menu

1. **Open ComfyUI**
2. **Click "Load"** in the menu
3. **Navigate** to the examples directory
4. **Select** the desired JSON file
5. **Configure and run**

## ⚙️ Configuration Requirements

### Required Files

Before running the examples, ensure you have:

1. **Model Files**:
   - `models/pixel3dmm/pixel3dmm_model.pth`
   - `models/flame/FLAME2020/generic_model.pkl`
   - `models/flame/FLAME_masks.pkl`

2. **Input Images**:
   - Place test images in `input/` directory
   - Supported formats: JPG, PNG, WEBP
   - Recommended resolution: 512x512 or higher

3. **Output Directories**:
   - `output/meshes/` (for exported 3D models)
   - `output/images/` (for rendered previews)

### Path Configuration

Update these paths in the workflows:

```json
// Model path in Pixel3DMMLoader
"model_path": "models/pixel3dmm/pixel3dmm_model.pth"

// Input image in LoadImage
"image": "your_face_photo.jpg"

// Output directory in MeshExporter
"output_directory": "output/meshes"
```

## 🎛️ Customization Options

### Quality Settings

**Fast Mode** (30 seconds):
```json
"reconstruction_quality": "fast",
"optimize_flame": false,
"optimization_steps": 50
```

**Balanced Mode** (1-2 minutes):
```json
"reconstruction_quality": "balanced", 
"optimize_flame": true,
"optimization_steps": 100
```

**High Quality Mode** (3-5 minutes):
```json
"reconstruction_quality": "high",
"optimize_flame": true,
"optimization_steps": 200
```

### Device Configuration

**Auto Detection**:
```json
"device": "auto",
"precision": "fp32"
```

**GPU Optimized**:
```json
"device": "cuda",
"precision": "fp16"
```

**CPU Only**:
```json
"device": "cpu",
"precision": "fp32"
```

### Export Options

**Basic OBJ Export**:
```json
"output_format": "obj",
"include_textures": false,
"scale_factor": 1.0
```

**Professional Export with Textures**:
```json
"output_format": "obj",
"include_textures": true,
"scale_factor": 1.0,
"center_mesh": true
```

## 🔧 Troubleshooting Examples

### Common Issues

#### ❌ "Model file not found"
**Solution**: Update the model path in Pixel3DMMLoader:
```json
"model_path": "/full/path/to/pixel3dmm_model.pth"
```

#### ❌ "Input image not found"
**Solution**: Update the image path in LoadImage:
```json
"image": "/full/path/to/your_image.jpg"
```

#### ❌ "Output directory not found"
**Solution**: Create the directory or update the path:
```bash
mkdir -p output/meshes
```

#### ❌ "CUDA out of memory"
**Solutions**:
1. Switch to CPU mode
2. Use FP16 precision
3. Reduce optimization steps
4. Use "fast" quality setting

### Performance Tips

1. **GPU Usage**: Always use GPU when available
2. **Batch Processing**: Process multiple images in sequence
3. **Memory Management**: Close other applications
4. **Quality vs Speed**: Choose appropriate quality settings

## 📚 Learning Path

### Beginner
1. Start with `basic_reconstruction.json`
2. Try different input images
3. Experiment with quality settings
4. Learn the basic node connections

### Intermediate
1. Use `advanced_optimization.json`
2. Understand UV and normal prediction
3. Experiment with optimization parameters
4. Try different export formats

### Advanced
1. Create custom workflows
2. Combine with other ComfyUI nodes
3. Implement batch processing
4. Develop custom configurations

## 🎯 Best Practices

### Input Images
- **Resolution**: 512x512 minimum, 1024x1024 optimal
- **Quality**: High-quality, well-lit face photos
- **Pose**: Frontal or near-frontal face orientation
- **Background**: Clean, uncluttered backgrounds work best

### Workflow Organization
- **Group nodes** logically (Input, Processing, Output)
- **Use descriptive names** for saved workflows
- **Document custom settings** in workflow descriptions
- **Save intermediate results** for debugging

### Performance Optimization
- **Use appropriate quality settings** for your use case
- **Monitor GPU memory usage** during processing
- **Save frequently** to avoid losing work
- **Test with small images** before processing large batches

## 🤝 Contributing Examples

We welcome community contributions of example workflows!

### Submission Guidelines
1. **Test thoroughly** before submitting
2. **Include documentation** explaining the workflow
3. **Use descriptive filenames** and node names
4. **Provide sample input/output** when possible

### Example Categories
- **Basic workflows** for beginners
- **Advanced techniques** for professionals
- **Specialized use cases** (animation, batch processing, etc.)
- **Integration examples** with other ComfyUI nodes

## 📞 Support

If you have questions about the examples:

1. **Check the main README.md** for general information
2. **Review the troubleshooting section** above
3. **Open an issue** on GitHub with your specific question
4. **Join our Discord** for community support

---

**Happy 3D reconstructing!** 🎉

These examples should get you started with creating amazing 3D face models using Pixel3DMM in ComfyUI.
