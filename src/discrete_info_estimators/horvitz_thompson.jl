export HorvitzThompson

"""
    HorvitzThompson <: DiscreteInfoEstimatorShannon
    HorvitzThompson(measure::Shannon = Shannon())


The `HorvitzThompson` estimator is used with [`information`](@ref) to compute the
discrete [`Shannon`](@ref) entropy according to [Horvitz1952](@citet).

# Description

The Horvitz-Thompson estimator of [`Shannon`](@ref) entropy is given by

```math
H_S^{HT} = -\\sum_{i=1}^M \\dfrac{p_i \\log(p_i) }{1 - (1 - p_i)^N},
```

where ``N`` is the sample size and ``M`` is the number of [`outcomes`](@ref).
Given the true probability ``p_i`` of the ``i``-th outcome, ``1 - (1 - p_i)^N`` is the
probability that the outcome appears at least once in a sample of size ``N``
[Arora2022](@cite). Dividing by this inclusion probability is a form of weighting, and
compensates for situations where certain outcomes have so low probabilities that they are
not often observed in a sample, for example in power-law distributions.
"""
Base.@kwdef struct HorvitzThompson{I <: InformationMeasure} <: DiscreteInfoEstimatorShannon{I}
    definition::I = Shannon()
end

function information(hest::HorvitzThompson{<:Shannon}, pest::ProbabilitiesEstimator, o::OutcomeSpace, x)
    (; definition) = hest
    probs = probabilities(pest, o, x)
    N = length(x)
    h = -sum(horwitz_thompsonᵢ(pᵢ, definition.base, N) for pᵢ in probs)
    return h
end

horwitz_thompsonᵢ(pᵢ, base, N) = (pᵢ * log(base, pᵢ)) / (1 - (1 - pᵢ)^N)
