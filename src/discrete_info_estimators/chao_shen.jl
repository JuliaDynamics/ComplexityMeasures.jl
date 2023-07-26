export ChaoShen

"""
    ChaoShen <: DiscreteInfoEstimator
    ChaoShen(measure::Shannon = Shannon())


The `ChaoShen` estimator computes the [`Shannon`](@ref) discrete entropy according
to Chao & Shen (2003)[^Chao2003].

## Description

This estimator is a modification of the [`HorvitzThompson`](@ref) estimator that
multiplies each plugin probability estimate by an estimate of sample coverage.
If ``f_1`` is the number of singletons (outcomes that occur only once) in a sample
of length ``N``, then the sample coverage is ``C = 1 - \\dfrac{f_1}{N}``. The Chao-Shen
estimator of Shannon entropy is then

```math
H_S^{HT} = -\\sum_i=1^M \\( \\dfrac{C p_i \\log(C p_i}{1 - (1 - C p_i)^N} \\),
```

where ``N`` is the sample size and ``M`` is the number of [`outcomes`](@ref). If
``f_1 = N``, then ``f_1`` is set to ``f_1 = N - 1`` to ensure positive entropy (Arora
et al., 2022)[^Arora2022].

[^Chao2003]:
    Chao, A., & Shen, T. J. (2003). Nonparametric estimation of Shannon’s index of
    diversity when there are unseen species in sample. Environmental and ecological
    statistics, 10(4), 429-443.
    [^Arora2022]:
[^Arora2022]:
    Arora, A., Meister, C., & Cotterell, R. (2022). Estimating the entropy of linguistic
    distributions. arXiv preprint arXiv:2204.01469.
"""
struct ChaoShen{I <: InformationMeasure} <: DiscreteInfoEstimator{I}
    measure::I
end
ChaoShen() = ChaoShen(Shannon())

function information(hest::ChaoShen{<:Shannon}, pest::ProbabilitiesEstimator, x)
    (; measure) = hest
    # Count singletons in the sample
    frequencies, outcomes = frequencies_and_outcomes(pest, x)
    f₁ = 0
    for f in frequencies
        if f == 1
            f₁ += 1
        end
    end
    N = length(x)
    if f₁ == N
        f₁ == N - 1
    end
    C = 1 - (f₁ / N)

    # Estimate Shannon entropy
    probs = Probabilities(frequencies)
    h = -sum(chao_shenᵢ(pᵢ, measure.base, N, C) for pᵢ in probs)
    return h
end

chao_shenᵢ(pᵢ, base, N, C) = (C*pᵢ * log(base, C*pᵢ)) / (1 - (1 - C*pᵢ)^N)
