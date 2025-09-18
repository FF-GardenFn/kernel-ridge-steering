# Experiment 4: Spectral-Enhanced ActAdd

Provenance: Mirrored from ../../experiments.md (as of 2025-09-18).

### Objective
Develop and validate improved activation steering using spectral guidance.

### Hypothesis
Spectral-weighted multi-layer injection outperforms fixed single-layer steering.

### Implementation Details

Spectral ActAdd Mechanism:

If steering vectors are computed from contrastive prompts (positive minus negative), then injecting them at spectrally unstable layers maximizes behavioral change, because activation differences encode directional information in representation space. The eigenvalue gap threshold (≤0.77) identifies layers where this directional information propagates most effectively through the network. Therefore, automatic layer selection based on spectral analysis eliminates the need for empirical hyperparameter search.

Adaptive Coefficient Logic:

If a layer has smaller eigenvalue gap, then it requires larger steering coefficient, because the inverse relationship between gap and controllability means unstable layers need stronger signals to achieve comparable effects. This adaptive scaling ensures uniform steering strength across layers despite varying spectral properties. Multi-layer injection then coordinates these scaled interventions, creating more robust behavioral modification than single-point injection.

### Expected Outcomes
- 20-30% improvement over baseline ActAdd
- More consistent effects across diverse prompts
- Reduced need for hyperparameter tuning
