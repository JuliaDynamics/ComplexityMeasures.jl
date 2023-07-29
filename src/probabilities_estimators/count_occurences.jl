export CountOccurrences

"""
    CountOccurrences()

A probabilities estimator based on straight-forward counting of distinct elements in
a univariate time series or multivariate dataset. This is the same as giving no
estimator to [`probabilities`](@ref).

## Outcome space

The outcome space is the unique sorted values of the input.
Hence, input `x` is needed for a well-defined [`outcome_space`](@ref).
"""
struct CountOccurrences <: ProbabilitiesEstimator end

function frequencies_and_outcomes(::CountOccurrences, x)
    z = copy(x)
    freqs = fasthist!(z)
    # notice that `z` is now sorted within `fasthist!` so we can skip sorting
    return freqs, unique!(z)
end

function probabilities_and_outcomes(est::CountOccurrences, x)
    freqs, outcomes = frequencies_and_outcomes(est, x)
    return Probabilities(freqs), outcomes
end

outcome_space(::CountOccurrences, x) = sort!(unique(x))
probabilities(::CountOccurrences, x) = probabilities(x)
frequencies(::CountOccurrences, x) = frequencies(x)

function probabilities(x)
    # Fast histograms code is in the `histograms` folder
    return Probabilities(frequencies(x))
end
function frequencies(x)
    return fasthist!(copy(x))
end
