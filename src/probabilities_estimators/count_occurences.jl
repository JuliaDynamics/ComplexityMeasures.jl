export CountOccurrences

"""
    CountOccurrences

A probabilities/entropy estimator based on straight-forward counting of distinct elements in
a univariate time series or multivariate dataset. This is the same as giving no
estimator to [`probabilities`](@ref).
"""
struct CountOccurrences <: ProbabilitiesEstimator end

probabilities(x::Array_or_Dataset, ::CountOccurrences) = probabilities(x)
