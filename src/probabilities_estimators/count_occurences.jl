export CountOccurrences

"""
    CountOccurrences()

A probabilities/entropy estimator based on straight-forward counting of distinct elements in
a univariate time series or multivariate dataset. This is the same as giving no
estimator to [`probabilities`](@ref).

The events in [`probabilities_and_events`](@ref) are the sorted unique values of
the input.
"""
struct CountOccurrences <: ProbabilitiesEstimator end

probabilities(x::Array_or_Dataset, ::CountOccurrences) = probabilities(x)
function probabilities(x::Array_or_Dataset)
    return Probabilities(fasthist(copy(x)))
end

function probabilities_and_events(x::Array_or_Dataset, ::CountOccurrences)
    z = copy(x)
    probs = Probabilities(fasthist(z))
    # notice that `z` is now sorted within `fasthist`!
    return probs, unique!(z)
end