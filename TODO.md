# Research TODO: Spectral-Guided Mechanistic Interpretability

##Phase 1: Immediate Validation 

### 1.1 Causal Validation of Spectral Predictions
- [ ] Implement activation patching for heads with gap ≤ 0.77
- [ ] Correlate eigenvalue gaps with causal importance scores
- [ ] Test "weakest link" principle: Does min(gap) head have max(importance)?
- [ ] Document correlation between spectral instability and causal necessity

### 1.2 Cross-Model Generalization
- [ ] Replicate eigenvalue analysis on Pythia-{410M, 1B, 2.8B}
- [ ] Test threshold consistency across model scales
- [ ] Validate on Llama-2-7B for architecture generalization
- [ ] Establish scaling laws for eigenvalue gap thresholds

### 1.3 Enhanced Steering Implementation
- [ ] Implement spectral-weighted ActAdd with automatic layer selection
- [ ] Compare performance: manual vs spectral-guided layer selection
- [ ] Develop multi-layer injection protocol based on gap distribution
- [ ] Quantify improvement over baseline Turner et al. method

## Phase 2: Mechanistic Understanding 

### 2.1 Feature Decomposition via Sparse Autoencoders
- [ ] Train SAEs specifically on unstable head outputs (gap ≤ 0.77)
- [ ] Compare feature sparsity: stable vs unstable heads
- [ ] Test hypothesis: unstable heads exhibit higher superposition
- [ ] Identify which features correlate with steerability

### 2.2 Circuit-Spectral Correspondence
- [ ] Map IOI circuit through eigenvalue landscape
- [ ] Identify if circuits preferentially route through unstable nodes
- [ ] Test whether known circuits contain spectral bottlenecks
- [ ] Develop spectral signatures for circuit types

### 2.3 Pre-Softmax Analysis
- [ ] Extract true QK^T scores before softmax
- [ ] Compare pre/post softmax eigenvalue structures
- [ ] Validate that key-based analysis captures relevant instabilities
- [ ] Refine threshold based on pre-softmax spectra

## Priority 3: Theoretical Extensions 

### 3.1 Mathematical Framework Development
- [ ] Formalize connection between KRR eigenvalues and controllability
- [ ] Derive theoretical bounds on steering effectiveness
- [ ] Prove weakest link principle from first principles
- [ ] Connect to control theory's controllability Gramian

### 3.2 Dynamic Stability Analysis
- [ ] Track eigenvalue evolution during inference
- [ ] Identify phase transitions in spectral properties
- [ ] Correlate with in-context learning emergence
- [ ] Develop real-time steering adaptation based on spectral state

### 3.3 Information-Theoretic Analysis
- [ ] Compute mutual information between gaps and steering success
- [ ] Develop entropy-based metrics for head instability
- [ ] Test information bottleneck hypothesis for unstable heads
- [ ] Quantify information flow through spectral gateways

## Priority 4: Applications 

### 4.1 Automated Steering System
- [ ] Build pipeline for automatic optimal layer identification
- [ ] Implement adaptive coefficient scaling based on gaps
- [ ] Create feature-aware steering using SAE decomposition
- [ ] Benchmark against existing steering methods

### 4.2 Safety Applications
- [ ] Test spectral signatures of deceptive circuits
- [ ] Identify instabilities correlating with harmful behaviors
- [ ] Develop spectral monitoring for runtime safety
- [ ] Create instability-based anomaly detection

### 4.3 Efficiency Optimizations
- [ ] Design architectures with controlled instabilities
- [ ] Test whether strategic instability improves few-shot learning
- [ ] Optimize attention head design for targeted steerability
- [ ] Develop spectral regularization for training

## Milestones

- **Week 2**: Causal validation complete, cross-model results available
- **Week 4**: SAE integration done, circuit mapping complete
- **Week 6**: Theoretical framework established, dynamic analysis functional
- **Week 8**: Full pipeline operational, paper draft ready

## Success Metrics

1. **Validation**: Causal importance correlation r > 0.7
2. **Generalization**: Threshold holds ±0.05 across models
3. **Performance**: 25% improvement over baseline steering
4. **Theory**: Closed-form prediction of steering from eigenvalues
5. **Application**: Automated system matches human expert selection


## Risk Mitigation

- **If threshold varies significantly**: Develop model-specific calibration
- **If SAEs don't converge**: Use alternative decomposition methods
- **If causal validation fails**: Refine spectral metrics beyond simple gap
- **If scaling breaks**: Focus on architecture-specific insights