export ChaoShen

"""
    ChaoShen <: DiscreteInfoEstimator
    ChaoShen(definition::Shannon = Shannon())

The `ChaoShen` estimator is used with [`information`](@ref) to compute the
discrete [`Shannon`](@ref) entropy according to Chao & Shen (2003)[^Chao2003].

## Description

This estimator is a modification of the [`HorvitzThompson`](@ref) estimator that
multiplies each plugin probability estimate by an estimate of sample coverage.
If ``f_1`` is the number of singletons (outcomes that occur only once) in a sample
of length ``N``, then the sample coverage is ``C = 1 - \\dfrac{f_1}{N}``. The Chao-Shen
estimator of Shannon entropy is then

```math
H_S^{CS} = -\\sum_{i=1}^M \\left( \\dfrac{C p_i \\log(C p_i)}{1 - (1 - C p_i)^N} \\right),
```

where ``N`` is the sample size and ``M`` is the number of [`outcomes`](@ref). If
``f_1 = N``, then ``f_1`` is set to ``f_1 = N - 1`` to ensure positive entropy (Arora
et al., 2022)[^Arora2022].

[^Chao2003]:
    Chao, A., & Shen, T. J. (2003). Nonparametric estimation of Shannon’s index of
    diversity when there are unseen species in sample. Environmental and ecological
    statistics, 10(4), 429-443.

[^Arora2022]:
    Arora, A., Meister, C., & Cotterell, R. (2022). Estimating the entropy of linguistic
    distributions. arXiv preprint arXiv:2204.01469.
"""
Base.@kwdef struct ChaoShen{I <: InformationMeasure} <: DiscreteInfoEstimator{I}
    definition::I = Shannon()
end

# Only works for real-count estimators.
function information(hest::ChaoShen{<:Shannon}, pest::ProbabilitiesEstimator, x)
    (; definition) = hest
    # Count singletons in the sample
    frequencies, outcomes = counts_and_outcomes(pest, x)
    f₁ = 0
    for f in frequencies
        if f == 1
            f₁ += 1
        end
    end
    # We should be using `N = length(x)`, but since some probabilities estimators
    # return pseudo counts, we need to consider those instead of counting actual
    # observations.
    N = sum(frequencies)
    if f₁ == N
        f₁ == N - 1
    end
    C = 1 - (f₁ / N)

    # Estimate Shannon entropy
    probs = Probabilities(frequencies)
    h = -sum(chao_shenᵢ(pᵢ, definition.base, N, C) for pᵢ in probs)
    return h
end

chao_shenᵢ(pᵢ, base, N, C) = (C*pᵢ * log(base, C*pᵢ)) / (1 - (1 - C*pᵢ)^N)
