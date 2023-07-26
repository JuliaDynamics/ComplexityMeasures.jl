export JackknifeEstimator

"""
    JackknifeEstimator <: DiscreteInfoEstimator
    JackknifeEstimator(measure::InformationMeasure = Shannon())

A generic estimator for discrete information measures using the jackknife principle.

## Description

### [`Shannon`](@ref) entropy

For [`Shannon`](@ref) entropy, the jackknife estimate is

```math
H_S^{J} = N H_S^{plugin} - \\ffrac{N-1}{N} \\sum_{i=1}^N {H_S^{plugin}}^{-\\{i\\}},
```

where ``N`` is the sample size, ``H_S^{plugin}`` is the plugin estimate of Shannon entropy,
and ``{H_S^{plugin}}^{-\\{i\\}}`` is the plugin estimate, but computed with the ``i``-th
sample left out.
"""
struct JackknifeEstimator{I <: InformationMeasure} <: DiscreteInfoEstimator{I}
    measure::I
end
JackknifeEstimator() = JackknifeEstimator(Shannon())

function information(hest::JackknifeEstimator{<:Shannon}, pest::ProbabilitiesEstimator, x)
    h_naive = information(PlugIn(hest.measure), pest, x)
    N = length(x)
    h_jackknifed = zeros(N)
    for i in eachindex(x)
        idxs = setdiff(1:N, i)
        xᵢ = @views x[idxs]
        h_jackknifed[i] = information(PlugIn(hest.measure), pest, xᵢ)
    end
    return N * h_naive - (N - 1)/N * sum(h_jackknifed)
end