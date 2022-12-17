export CountOccurrences

"""
    CountOccurrences(x)

A probabilities/entropy estimator based on straight-forward counting of distinct elements in
a univariate time series or multivariate dataset. This is the same as giving no
estimator to [`probabilities`](@ref).

## Outcome space
The outcome space is the unique sorted values of the input.
Hence, input `x` is needed for a well-defined outcome space.
"""
struct CountOccurrences{X} <: ProbabilitiesEstimator
    x::X
end

function probabilities_and_outcomes(::CountOccurrences, x::Array_or_Dataset)
    z = copy(x)
    probs = Probabilities(fasthist!(z))
    # notice that `z` is now sorted within `fasthist!` so we can skip sorting
    return probs, unique!(z)
end

outcome_space(est::CountOccurrences) = sort!(unique(est.x))

probabilities(::CountOccurrences, x::Array_or_Dataset) = probabilities(x)
function probabilities(x::Array_or_Dataset)
    # Fast histograms code is in the `histograms` folder
    return Probabilities(fasthist!(copy(x)))
end
