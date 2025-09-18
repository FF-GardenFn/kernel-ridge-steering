# Experiment 3: Circuit-Spectral Mapping

Provenance: Mirrored from ../../experiments.md (as of 2025-09-18).

### Objective
Map known computational circuits through spectral landscape to identify if circuits exploit unstable nodes.

### Hypothesis
Computational circuits preferentially route through spectrally unstable heads, which serve as controllable bottlenecks.

### Protocol

Circuit-Spectral Mapping Logic:

If computational circuits route through attention heads, then heads with smaller eigenvalue gaps should serve as critical bottlenecks, because spectral instability indicates high sensitivity to perturbations. Known circuits like IOI (Indirect Object Identification) rely on information routing through specific attention patterns, and these routing points become controllable when spectrally unstable. Therefore, intervening at heads with gaps ≤ 0.77 along a circuit path should maximally disrupt circuit function.

Bottleneck Identification Process:

1. If a head participates in a known circuit AND has gap ≤ 0.77, then classify as spectral bottleneck, because the intersection of functional importance and mathematical instability creates maximal leverage.
2. If we intervene at spectral bottlenecks, then circuit performance degrades more than intervening at stable nodes, because unstable nodes propagate perturbations through their high condition number.
3. If circuit fragility correlates with minimum gap along path, then spectral analysis predicts circuit robustness, because the weakest link principle applies to information flow through transformer layers.

### Expected Outcomes
- Critical circuit nodes have gaps < 0.77 in >60% of cases
- Intervention at spectral bottlenecks disrupts circuit function
- Circuit fragility correlates with minimum gap along path
