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
struct CountOccurrences <: OutcomeSpace end

function counts_and_outcomes(::CountOccurrences, x)
    z = copy(x)
    cts = fasthist!(z)
    # notice that `z` is now sorted within `frequencies!` so we can skip sorting
    return cts, unique!(z)
end

function probabilities_and_outcomes(est::CountOccurrences, x)
    cts, outcomes = counts_and_outcomes(est, x)
    return Probabilities(cts), outcomes
end

outcome_space(::CountOccurrences, x) = sort!(unique(x))
probabilities(::CountOccurrences, x) = probabilities(x)
counts(::CountOccurrences, x) = counts(x)

function probabilities(x)
    # Fast histograms code is in the `histograms` folder
    return Probabilities(counts(x))
end
function counts(x)
    return fasthist!(copy(x))
end


encoded_space_cardinality(o::CountOccurrences, x) = length(x)
