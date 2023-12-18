export UniqueElements

"""
    UniqueElements()

An [`OutcomeSpace`](@ref) based on straight-forward counting of distinct elements in
a univariate time series or multivariate dataset. This is the same as giving no
estimator to [`probabilities`](@ref).

## Outcome space

The outcome space is the unique sorted values of the input.
Hence, input `x` is needed for a well-defined [`outcome_space`](@ref).

## Implements

- [`codify`](@ref). Used for encoding inputs where ordering matters (e.g. time series).
"""
struct UniqueElements <: CountBasedOutcomeSpace end

is_counting_based(o::UniqueElements) = true
counts(::UniqueElements, x) = counts(x)
function counts_and_outcomes(::UniqueElements, x)
    z = copy(x)
    cts = fasthist!(z)
    # notice that `z` is now sorted within `fasthist!` so we can skip sorting
    outs = unique!(z)
    cts = Counts(cts, (outs, ))
    return cts, outcomes(cts)
end

outcome_space(::UniqueElements, x) = sort!(unique(x))

function codify(o::UniqueElements, x)
    encoding = UniqueElementsEncoding(x)
    encode.(Ref(encoding), x)
end
