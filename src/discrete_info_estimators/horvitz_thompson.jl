export HorvitzThompson

"""
    HorvitzThompson <: DiscreteInfoEstimator
    HorvitzThompson(measure::Shannon = Shannon())


The `HorvitzThompson` estimator computes the [`Shannon`](@ref) discrete entropy according
to Horvitz and Thompson (1952)[^Horvitz1952].

# Description

The Horvitz-Thompson estimator of Shannon entropy is given by

```math
H_S^{HT} = -\\sum_i=1^M \\( \\dfrac{p_i \\log(p_i}{1 - (1 - p_i)^N} \\),
```

where ``N`` is the sample size and ``M`` is the number of [`outcomes`](@ref).

[^Horvitz1952]:
    Horvitz, D. G., & Thompson, D. J. (1952). A generalization of sampling without
    replacement from a finite universe. Journal of the American statistical Association,
    47(260), 663-685.

"""
struct HorvitzThompson{I <: InformationMeasure} <: DiscreteInfoEstimator{I}
    measure::I
end

function information(hest::HorvitzThompson{<:Shannon}, pest::ProbabilitiesEstimator, x)
    (; measure) = hest
    probs = probabilities(pest, x)
    N = length(x)
    h = -sum(horwitz_thompsonᵢ(pᵢ, measure.base, N) for pᵢ in probs)
    return h
end

horwitz_thompsonᵢ(pᵢ, base, N) = (pᵢ * log(base, pᵢ)) / (1 - (1 - pᵢ)^N)
