export SampleCoverage

"""
    SampleCoverage <: ProbabilitiesEstimator
    SampleCoverage(outcome_space::OutcomeSpaceModel)

A probabilities estimator based on the Good-Turing correction, which uses the
observation that the number of singletons (i.e. outcomes with frequency 1) contains
information about unobserved outcomes.



[^Chao2003]:
    Chao, A., & Shen, T. J. (2003). Nonparametric estimation of Shannon’s index of
    diversity when there are unseen species in sample. Environmental and ecological
    statistics, 10(4), 429-443.

is used with [`probabilities`](@ref) to estimates
probabilities over the given [`OutcomeSpaceModel`](@ref) using empirical plug-in estimates
(also called "maximum likelihood estimation"; hence "RelativeAmount").
"""
struct SampleCoverage{O <: OutcomeSpaceModel} <: ProbabilitiesEstimator
    outcome_space::O
end

function probabilities_and_outcomes(est::SampleCoverage, x)
    # Count singletons in the sample
    freqs, outcomes = frequencies_and_outcomes(outcome_space(est), x)
    f₁ = 0
    for f in freqs
        if f == 1
            f₁ += 1
        end
    end
    N = length(x)
    if f₁ == N
        f₁ == N - 1
    end
    C = 1 - (f₁ / N)
    return (freqs ./ N) .* C
end
