# Experiment 2: Sparse Autoencoder Feature Analysis

Provenance: Mirrored from ../../experiments.md (as of 2025-09-18).

### Objective
Identify interpretable features in spectrally unstable attention heads and test superposition hypothesis.

### Hypothesis
Unstable heads (gap ≤ 0.77) encode more polysemantic features, creating mathematical instability through superposition.

### Expected Outcomes
- Unstable heads: 30-50% higher polysemanticity
- More features active simultaneously in unstable heads
- Lower interpretability scores for unstable head features
- Higher reconstruction loss indicating compression

### Feature-Guided Steering

SAE-Enhanced Steering Logic:

If we decompose steering vectors through SAE features, then we can selectively amplify interpretable components, because SAEs learn sparse representations where individual features correspond to semantic concepts. These features are linearly separable in the learned dictionary space, which allows targeted modification without corrupting unrelated representations. Therefore, amplifying features associated with desired behaviors while suppressing others creates more precise steering than raw vector addition.
