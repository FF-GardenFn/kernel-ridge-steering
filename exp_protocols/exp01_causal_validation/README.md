# Experiment 1: Causal Validation of Spectral Predictions

Provenance: Mirrored from ../../experiments.md (as of 2025-09-18).

### Objective
Establish causal relationship between eigenvalue gaps and steering effectiveness through activation patching.

### Hypothesis
Attention heads with eigenvalue gaps ≤ 0.77 are not merely correlated with steering effectiveness but are causally necessary for behavioral modification.

### Expected Outcomes
- Unstable heads show >2x restoration compared to stable heads
- Correlation between (1/gap) and causal importance: r > 0.7
- Validates weakest link principle causally

### Analysis Plan
1. Statistical significance testing via bootstrap
2. Cross-validation across prompt types
3. Ablation studies removing individual heads
