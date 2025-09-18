# Experimental Validation Protocols: Spectral-Guided Mechanistic Interpretability

## Experiment 1: Causal Validation of Spectral Predictions

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

---

## Experiment 2: Sparse Autoencoder Feature Analysis

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

**SAE-Enhanced Steering Logic:**

If we decompose steering vectors through SAE features, then we can selectively amplify interpretable components, because SAEs learn sparse representations where individual features correspond to semantic concepts. These features are linearly separable in the learned dictionary space, which allows targeted modification without corrupting unrelated representations. Therefore, amplifying features associated with desired behaviors while suppressing others creates more precise steering than raw vector addition.

---

## Experiment 3: Circuit-Spectral Mapping

### Objective
Map known computational circuits through spectral landscape to identify if circuits exploit unstable nodes.

### Hypothesis
Computational circuits preferentially route through spectrally unstable heads, which serve as controllable bottlenecks.

### Protocol

**Circuit-Spectral Mapping Logic:**

If computational circuits route through attention heads, then heads with smaller eigenvalue gaps should serve as critical bottlenecks, because spectral instability indicates high sensitivity to perturbations. Known circuits like IOI (Indirect Object Identification) rely on information routing through specific attention patterns, and these routing points become controllable when spectrally unstable. Therefore, intervening at heads with gaps ≤ 0.77 along a circuit path should maximally disrupt circuit function.

**Bottleneck Identification Process:**

1. If a head participates in a known circuit AND has gap ≤ 0.77, then classify as spectral bottleneck, because the intersection of functional importance and mathematical instability creates maximal leverage.

2. If we intervene at spectral bottlenecks, then circuit performance degrades more than intervening at stable nodes, because unstable nodes propagate perturbations through their high condition number.

3. If circuit fragility correlates with minimum gap along path, then spectral analysis predicts circuit robustness, because the weakest link principle applies to information flow through transformer layers.

### Expected Outcomes
- Critical circuit nodes have gaps < 0.77 in >60% of cases
- Intervention at spectral bottlenecks disrupts circuit function
- Circuit fragility correlates with minimum gap along path

---

## Experiment 4: Spectral-Enhanced ActAdd

### Objective
Develop and validate improved activation steering using spectral guidance.

### Hypothesis
Spectral-weighted multi-layer injection outperforms fixed single-layer steering.



### Implementation Details

**Spectral ActAdd Mechanism:**

If steering vectors are computed from contrastive prompts (positive minus negative), then injecting them at spectrally unstable layers maximizes behavioral change, because activation differences encode directional information in representation space. The eigenvalue gap threshold (≤0.77) identifies layers where this directional information propagates most effectively through the network. Therefore, automatic layer selection based on spectral analysis eliminates the need for empirical hyperparameter search.

**Adaptive Coefficient Logic:**

If a layer has smaller eigenvalue gap, then it requires larger steering coefficient, because the inverse relationship between gap and controllability means unstable layers need stronger signals to achieve comparable effects. This adaptive scaling ensures uniform steering strength across layers despite varying spectral properties. Multi-layer injection then coordinates these scaled interventions, creating more robust behavioral modification than single-point injection.

### Expected Outcomes
- 20-30% improvement over baseline ActAdd
- More consistent effects across diverse prompts
- Reduced need for hyperparameter tuning

---

## Experiment 5: Dynamic Spectral Monitoring

### Objective
Track eigenvalue evolution during inference and adapt steering in real-time.

### Hypothesis
Spectral properties change dynamically during generation, requiring adaptive intervention.

### Protocol

**Dynamic Monitoring Logic:**

If spectral properties evolve during generation, then steering effectiveness varies across timesteps, because attention patterns shift as context accumulates. Eigenvalue gaps can transition from stable to unstable as the model processes different token types, creating windows of opportunity for intervention. Therefore, monitoring spectral evolution and adapting steering strength in real-time maintains consistent control throughout generation.

**Adaptive Intervention Process:**

1. If eigenvalue gaps decrease below threshold during generation, then emerging instability signals increased controllability, because the model enters a more malleable computational state.

2. If spectral phase transitions occur (sudden gap changes), then these mark behavioral regime shifts, because eigenvalue structure reflects underlying computational modes.

3. If we adapt steering strength based on current spectral state, then maintain consistent influence despite changing controllability, because dynamic adjustment compensates for varying sensitivity.

### Expected Outcomes
- Identification of spectral phase transitions during generation
- Correlation between instability emergence and behavioral shifts
- Improved steering consistency through dynamic adaptation

---

## Experiment 6: Cross-Architecture Validation

### Objective
Test spectral framework generalization across model families.

### Hypothesis
The eigenvalue gap threshold (≤0.77) represents a universal property of transformer steering, invariant to specific architectures.

### Protocol

**Cross-Architecture Validation Logic:**

If the eigenvalue gap threshold represents fundamental mathematical properties of attention, then it should generalize across transformer architectures, because the kernel ridge regression formulation applies universally to dot-product attention. Different model families (GPT, Pythia, LLaMA, Mistral) share the same attention mechanism despite varying in scale and training, so spectral properties should exhibit consistent patterns. Therefore, the 0.77 threshold should hold within ±0.05 across architectures.

**Universality Testing Process:**

1. If optimal thresholds cluster around 0.77 across models, then confirms universal spectral principle, because mathematical properties transcend implementation details.

2. If correlation between gaps and steering effectiveness remains consistent, then validates that spectral instability creates controllability regardless of model specifics.

3. If scaling laws emerge relating threshold to model properties (size, heads, dimensions), then provides predictive framework for new architectures, because systematic relationships indicate fundamental principles.

### Expected Outcomes
- Threshold consistency: σ(threshold) < 0.05 across models
- Scaling law: threshold = f(model_size, n_heads, d_model)
- Universal correlation pattern maintained

---
see [individual expriments](./exp_protocols) for more details. (this file will eventually be deleted or summary of experiments results)

