export MillerMadow

"""
    MillerMadow <: DiscreteInfoEstimatorShannon
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
[Paninski2003](@citet)), and `N` is the number of observations (the number of encoded
points, e.g. the number of ordinal patterns for [`OrdinalPatterns`](@ref)). The
correction ``(m - 1)/2N`` is in nats; for an entropy in base ``b`` it is divided by
``\\log(b)``.
"""
Base.@kwdef struct MillerMadow{I <: InformationMeasure} <: DiscreteInfoEstimatorShannon{I}
    definition::I = Shannon()
end

function information(hest::MillerMadow{<:Shannon}, pest::ProbabilitiesEstimator, o::OutcomeSpace, x)
    # The number of observations is the number of encoded points, which is smaller than
    # `length(x)` for outcome spaces that embed `x` (e.g. `OrdinalPatterns`).
    N = is_counting_based(o) ? sum(counts(o, x)) : encoded_space_cardinality(o, x)
    probs = allprobabilities(pest, o, x)
    # Estimate of the number of bins with nonzero pN-probability; here estimated as
    # in Paninski (2003)
    m̂ = count(probs .> 0.0)
    h_naive = information(PlugIn(hest.definition), probs)
    # The correction (m̂ - 1) / 2N is in nats; convert it to the unit of the entropy.
    return h_naive + convert_logunit((m̂ - 1) / (2 * N), ℯ, hest.definition.base)
end
