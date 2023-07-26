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
Given the true probability ``p_i`` of the ``i``-th outcome, ``1 - (1 - p_i)^N`` is the
probability that the outcome appears at least once in a sample of size ``N`` (Arora et al.,
2022). Dividing by this inclusion probability is a form of weighting, and compensates
for situations where certain outcomes have so low probabilities that they are not
often observed in a sample, for example in power-law distributions.

[^Horvitz1952]:
    Horvitz, D. G., & Thompson, D. J. (1952). A generalization of sampling without
    replacement from a finite universe. Journal of the American statistical Association,
    47(260), 663-685.
[^Arora2022]:
    Arora, A., Meister, C., & Cotterell, R. (2022). Estimating the entropy of linguistic
    distributions. arXiv preprint arXiv:2204.01469.
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
