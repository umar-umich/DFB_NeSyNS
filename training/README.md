# NeSyDeFake - Modular Architecture

## Overview
This is the refactored, modular version of NeSyDeFake (Neural-Symbolic Deepfake Detection with Causal Discovery). The code has been split into logical, reusable modules for better maintainability, ablation studies, and research flexibility.

## Directory Structure

```
training/
├── detectors/
│   └── nesydefake_hybrid.py          # Main orchestrator (~300 lines)
│
├── networks/
│   ├── __init__.py                   # Package exports
│   │
│   ├── foundation_models/            # Module 1: Feature Extractors
│   │   ├── __init__.py
│   │   ├── temporal_extractor.py     # VideoMAE-based
│   │   ├── spatial_extractor.py      # CLIP-based
│   │   └── frequency_extractor.py    # SRM+ResNet
│   │
│   ├── fusion/                       # Multi-modal Fusion
│   │   ├── __init__.py
│   │   ├── multimodal_fusion.py      # Concat/Attention/Weighted
│   │   └── cross_modal_attention.py  # Cross-modal attention
│   │
│   ├── causal/                       # Module 3: Causal Discovery
│   │   ├── __init__.py
│   │   ├── latent_encoder.py         # Latent variable encoding
│   │   ├── concept_extractor.py      # Semantic concepts
│   │   ├── causal_learner.py         # DAG structure learning
│   │   ├── causal_reasoner.py        # Causal inference
│   │   └── causal_discovery.py       # Main causal module
│   │
│   └── classifiers/                  # Module 5: Classification
│       ├── __init__.py
│       ├── multitask_head.py         # Multi-task classifier
│       └── sparse_autoencoder.py     # Sparse features (optional)
│
└── utils/
    └── semantic_grounding.py         # Module 2: DeepFace integration
```

## Module Descriptions

### 1. Foundation Models (`networks/foundation_models/`)
**Purpose:** Extract features from different modalities

- **TemporalFeatureExtractor**: VideoMAE for temporal dynamics
- **SpatialFeatureExtractor**: CLIP for spatial appearance
- **FrequencyFeatureExtractor**: SRM+ResNet for frequency artifacts

**Key Features:**
- Pretrained model loading
- Selective freezing for transfer learning
- Modular replacement of backbones

### 2. Fusion (`networks/fusion/`)
**Purpose:** Combine multi-modal features

- **MultiModalFusion**: Main fusion module (concat/attention/weighted)
- **CrossModalAttention**: Cross-attention between modalities

**Key Features:**
- Multiple fusion strategies
- Learnable fusion weights
- Dimension projection

### 3. Causal Discovery (`networks/causal/`)
**Purpose:** Learn and reason about causal structures

- **LatentVariableEncoder**: Extract latent representations
- **SemanticConceptExtractor**: Learn interpretable concepts
- **StructuralCausalCircuits**: Learn DAG structure
- **CausalReasoner**: Perform causal inference
- **CausalDiscoveryModule**: Orchestrate causal components

**Key Features:**
- Differentiable DAG learning
- Acyclicity constraints
- Interventional reasoning
- Concept interpretability

### 4. Semantic Grounding (`utils/semantic_grounding.py`)
**Purpose:** Extract explicit semantic concepts using DeepFace

**Key Features:**
- Age, gender, emotion, race extraction
- Face detection and analysis
- Feature fusion with implicit features

### 5. Classifiers (`networks/classifiers/`)
**Purpose:** Final classification and feature learning

- **MultiTaskHead**: Multi-task learning (classification, uncertainty, etc.)
- **SparseAutoencoder**: Learn monosemantic features

**Key Features:**
- Task-specific heads
- Shared representations
- Sparse feature discovery

## Usage

### Basic Usage

```python
from detectors import DETECTOR

# Load config
config = load_yaml_config('config/nesydefake_hybrid.yaml')

# Initialize detector
detector = DETECTOR(config, module_name='nesydefake_hybrid')

# Forward pass
pred_dict = detector(data_dict)
```

### Ablation Studies

#### 1. Disable Causal Module
```yaml
# In config file
causal_module:
  enabled: false
```

#### 2. Disable Semantic Grounding
```yaml
semantic_grounding:
  enabled: false
```

#### 3. Change Fusion Strategy
```yaml
fusion:
  type: attention  # Options: concat, attention, weighted
```

#### 4. Use Different Backbone
```python
# In temporal_extractor.py
from transformers import VideoMAEModel
# Replace with any video model
self.backbone = YourVideoModel.from_pretrained(...)
```

### Progressive Training

The config supports multi-phase training:

```yaml
training_phases:
  phase1:
    epochs: [0, 5]
    freeze_modules: ['foundation_models']
    train_modules: ['fusion', 'classifier']
    
  phase2:
    epochs: [5, 10]
    train_modules: ['all']
    
  phase3:
    epochs: [10, 20]
    enable_causal: true
```

## Integration with DeepfakeBench

### File Placement

1. **Copy detector:**
   ```bash
   cp detectors/nesydefake_hybrid.py YOUR_REPO/training/detectors/
   ```

2. **Copy networks:**
   ```bash
   cp -r networks/ YOUR_REPO/training/
   ```

3. **Update semantic grounding:**
   ```bash
   # If not already present
   cp utils/semantic_grounding.py YOUR_REPO/training/utils/
   ```

### Config File
Place your config at: `YOUR_REPO/training/config/detector/nesydefake_hybrid.yaml`

## Key Advantages of Modular Design

### 1. **Easy Ablation Studies**
- Toggle modules on/off via config
- Swap components independently
- Compare different architectures

### 2. **Better Code Organization**
- Each module has clear responsibility
- Easier to debug and test
- Better version control

### 3. **Plug-and-Play Components**
- Replace VideoMAE with other video models
- Try different fusion strategies
- Experiment with causal discovery algorithms

### 4. **Scalability**
- Add new feature extractors easily
- Extend with new tasks
- Support multi-dataset training

### 5. **Research Flexibility**
- Quick prototyping of new ideas
- Easy comparison of approaches
- Clear baseline for improvements

## Example Ablation Experiments

### Experiment 1: Feature Extractor Ablation
```yaml
# Test temporal only
foundation_models:
  spatial:
    freeze_backbone: true  # Freeze or disable
  frequency:
    freeze_backbone: true
```

### Experiment 2: Fusion Strategy Comparison
```yaml
# Test different fusion methods
fusion:
  type: concat  # Then: attention, weighted
```

### Experiment 3: Causal Discovery Impact
```yaml
# Compare with/without causal reasoning
causal_module:
  enabled: false  # Then: true
```

### Experiment 4: Semantic Grounding Contribution
```yaml
# Test explicit vs implicit concepts
semantic_grounding:
  enabled: false  # Then: true
  extract_age: true
  extract_gender: true
  extract_emotion: true
```

## Performance Optimization

### 1. **Memory Efficiency**
- Freeze pretrained models initially
- Use gradient checkpointing for VideoMAE
- Enable mixed precision training

### 2. **Training Speed**
```yaml
# In config
mixed_precision: true
grad_clip: 1.0

# Batch processing for semantic grounding
semantic_grounding:
  batch_process: true
```

### 3. **Multi-GPU Support**
The modular design works seamlessly with DataParallel and DistributedDataParallel.

## Testing Individual Modules

```python
# Test temporal extractor
from networks.foundation_models import TemporalFeatureExtractor

temporal = TemporalFeatureExtractor(config)
features = temporal(video_clips)
print(f"Temporal features: {features.shape}")

# Test causal module
from networks.causal import CausalDiscoveryModule

causal = CausalDiscoveryModule(config)
violation_score, dag, concepts = causal(features, return_graph=True)
print(f"Learned DAG:\n{dag}")
```

## Troubleshooting

### Import Errors
Make sure `training/` is in your Python path:
```python
import sys
sys.path.append('/path/to/DeepfakeBench/training')
```

### CUDA Out of Memory
- Reduce batch size
- Freeze more backbone layers
- Disable gradient checkpointing on causal module

### DeepFace Issues
- Ensure face detection backend is installed
- Set `enforce_detection: false` for robustness
- Use batch processing: `batch_process: true`

## Citation

If you use this code, please cite:

```bibtex
@article{nesydefake2024,
  title={NeSyDeFake: Neural-Symbolic Deepfake Detection with Causal Discovery},
  author={Your Name},
  journal={Conference},
  year={2024}
}
```

## License

[Your License Here]

## Contributing

To add a new module:

1. Create new file in appropriate `networks/` subdirectory
2. Update `__init__.py` to export your module
3. Add config section in YAML
4. Update main detector to use your module

## Contact

For questions or issues, please open a GitHub issue or contact [your email].
