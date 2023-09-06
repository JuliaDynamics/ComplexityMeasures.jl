export CountOccurrences

"""
    CountOccurrences()

An [`OutcomeSpace`](@ref) based on straight-forward counting of distinct elements in
a univariate time series or multivariate dataset. This is the same as giving no
estimator to [`probabilities`](@ref).

## Outcome space

The outcome space is the unique sorted values of the input.
Hence, input `x` is needed for a well-defined [`outcome_space`](@ref).

## Implements

- [`symbolize`](@ref). Used for encoding inputs where ordering matters (e.g. time series).
"""
struct CountOccurrences <: CountBasedOutcomeSpace end

is_counting_based(o::CountOccurrences) = true
counts(::CountOccurrences, x) = counts(x)
function counts_and_outcomes(::CountOccurrences, x)
    z = copy(x)
    cts = fasthist!(z)
    # notice that `z` is now sorted within `frequencies!` so we can skip sorting
    return Counts(cts), unique!(z)
end

outcome_space(::CountOccurrences, x) = sort!(unique(x))
probabilities(::CountOccurrences, x) = probabilities(x)

symbolize(o::CountOccurrences, x::AbstractVector) = x
