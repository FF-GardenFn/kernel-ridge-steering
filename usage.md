## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/FF-GardenFn/kernel-ridge-steering.git
cd Kernel-ridge-steering

# Install dependencies
pip install torch transformers transformer-lens matplotlib numpy scipy
```

### Basic Usage

```python
from implementation.krr_steering_fixed import SpectralAnalyzer, SteeringValidator

# Analyze spectral properties
analyzer = SpectralAnalyzer(model='gpt2')
gaps = analyzer.compute_eigenvalue_gaps(layer=8)

# Identify optimal steering layers
steerable_layers = analyzer.find_steerable_layers(threshold=0.77)

# Validate steering effectiveness
validator = SteeringValidator(model='gpt2')
effects = validator.measure_steering_effects(layers=steerable_layers)
```