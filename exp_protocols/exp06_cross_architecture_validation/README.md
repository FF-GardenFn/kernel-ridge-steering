# Experiment 6: Cross-Architecture Validation

Provenance: Mirrored from ../../experiments.md (as of 2025-09-18).

### Objective
Test spectral framework generalization across model families.

### Hypothesis
The eigenvalue gap threshold (≤0.77) represents a universal property of transformer steering, invariant to specific architectures.

### Protocol

Cross-Architecture Validation Logic:

If the eigenvalue gap threshold represents fundamental mathematical properties of attention, then it should generalize across transformer architectures, because the kernel ridge regression formulation applies universally to dot-product attention. Different model families (GPT, Pythia, LLaMA, Mistral) share the same attention mechanism despite varying in scale and training, so spectral properties should exhibit consistent patterns. Therefore, the 0.77 threshold should hold within ±0.05 across architectures.

Universality Testing Process:

1. If optimal thresholds cluster around 0.77 across models, then confirms universal spectral principle, because mathematical properties transcend implementation details.
2. If correlation between gaps and steering effectiveness remains consistent, then validates that spectral instability creates controllability regardless of model specifics.
3. If scaling laws emerge relating threshold to model properties (size, heads, dimensions), then provides predictive framework for new architectures, because systematic relationships indicate fundamental principles.

### Expected Outcomes
- Threshold consistency: σ(threshold) < 0.05 across models
- Scaling law: threshold = f(model_size, n_heads, d_model)
- Universal correlation pattern maintained
