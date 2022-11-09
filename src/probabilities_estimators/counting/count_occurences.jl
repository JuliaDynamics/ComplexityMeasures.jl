export CountOccurrences

"""
    CountOccurrences()

A probabilities/entropy estimator based on straight-forward counting of distinct elements in
a univariate time series or multivariate dataset. This is the same as giving no
estimator to [`probabilities`](@ref).

## Outcomes

The outcomes `Ω` for `CountOccurrences` are the unique sorted values of the input. Use
[`probabilities_and_outcomes`](@ref) to obtain these together with the probabilities.

See also: [`probabilities_and_outcomes`](@ref).
"""
struct CountOccurrences <: ProbabilitiesEstimator end

probabilities(x::Array_or_Dataset, ::CountOccurrences) = probabilities(x)
function probabilities(x::Array_or_Dataset)
    return Probabilities(fasthist!(copy(x)))
end

outcomes(x, ::CountOccurrences) = sort(unique(x))

function probabilities_and_outcomes(x::Array_or_Dataset, ::CountOccurrences)
    z = copy(x)
    probs = Probabilities(fasthist!(z))
    # notice that `z` is now sorted within `fasthist!` so we can skip sorting
    return probs, unique!(z)
end
