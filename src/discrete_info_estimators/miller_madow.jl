export MillerMadow

"""
    MillerMadow <: DiscreteInfoEstimator
    MillerMadow(measure::Shannon = Shannon())

The `MillerMadow` estimator is used with [`information`](@ref) to compute the
discrete [`Shannon`](@ref) entropy according to [Miller1955](@citet).

# Description

The Miller-Madow estimator of Shannon entropy is given by

```math
H_S^{MM} = H_S^{plugin} + \\dfrac{m - 1}{2N},
```

where ``H_S^{plugin}`` is the Shannon entropy estimated using the [`PlugIn`](@ref)
estimator, `m` is the number of bins with nonzero probability (as defined in
[Paninski2003](@citet)), and `N` is the number of observations.
"""
Base.@kwdef struct MillerMadow{I <: InformationMeasure} <: DiscreteInfoEstimator{I}
    definition::I = Shannon()
end

function information(hest::MillerMadow{<:Shannon}, pest::ProbabilitiesEstimator, x)
    N = length(x)
    probs = allprobabilities(pest, x)
    # Estimate of the number of bins with nonzero pN-probability; here estimated as
    # in Paninski (2003)
    m̂ = count(probs .> 0.0)
    h_naive = information(PlugIn(hest.definition), probs)
    return h_naive + (m̂ - 1)/(2 * N)
end
