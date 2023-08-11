export OutcomeSpace
export outcome_space
export total_outcomes
export missing_outcomes
export outcomes
export counts
export counts_and_outcomes
export frequencies
export allcounts_and_outcomes

"""
    OutcomeSpace

The supertype for all outcome space models.

## Description

In ComplexityMeasures.jl, an outcome space model defines a set of possible outcomes
``\\Omega = \\{\\omega_1, \\omega_2, \\ldots, \\omega_L \\}`` (some form of
discretization). It also defines a set of rules for mapping input data to
to each outcome ``\\omega_i`` (i.e. [encoding](@ref encodings)/discretizing).

## Implementations

| Outcome space                               | Principle                    | Input data                | Counting-compatible |
| :------------------------------------------ | :--------------------------- | :------------------------ | :------------------ |
| [`CountOccurrences`](@ref)                  | Count of unique elements     | `Any`                     | ✔                  |
| [`ValueHistogram`](@ref)                    | Binning (histogram)          | `Vector`, `StateSpaceSet` | ✔                  |
| [`TransferOperator`](@ref)                  | Binning (transfer operator)  | `Vector`, `StateSpaceSet` | ✖                  |
| [`NaiveKernel`](@ref)                       | Kernel density estimation    | `StateSpaceSet`           | ✖                  |
| [`SymbolicPermutation`](@ref)               | Ordinal patterns             | `Vector`, `StateSpaceSet` | ✔                  |
| [`SymbolicWeightedPermutation`](@ref)       | Ordinal patterns             | `Vector`, `StateSpaceSet` | ✖                  |
| [`SymbolicAmplitudeAwarePermutation`](@ref) | Ordinal patterns             | `Vector`, `StateSpaceSet` | ✖                  |
| [`SpatialSymbolicPermutation`](@ref)        | Ordinal patterns in space    | `Array`                   | ✔                  |
| [`Dispersion`](@ref)                        | Dispersion patterns          | `Vector`                  | ✔                  |
| [`SpatialDispersion`](@ref)                 | Dispersion patterns in space | `Array`                   | ✔                  |
| [`Diversity`](@ref)                         | Cosine similarity            | `Vector`                  | ✖                  |
| [`WaveletOverlap`](@ref)                    | Wavelet transform            | `Vector`                  | ✖                  |
| [`PowerSpectrum`](@ref)                     | Fourier transform            | `Vector`                  | ✖                  |

In the column "input data" it is assumed that the `eltype` of the input is `<: Real`.

## Usage

Outcome spaces are used as input to

- [`probabilities`](@ref)/[`probabilities_and_outcomes`](@ref), for computing probability
    mass functions.
- [`allprobabilities`](@ref)/[`allprobabilities_and_outcomes`](@ref), for computing
    probability mass functions, guaranteeing that also zero-probability outcomes are
    are included.
- [`outcome_space`](@ref), which returns the elements of the outcome space.
- [`total_outcomes`](@ref), which returns the cardinality of the outcome space.
- [`counts`](@ref)/[`counts_and_outcomes`](@ref), for obtaining raw counts instead
    of probabilities (only for counting-compatible outcome spaces).

# Counting-compatible vs. non-counting compatible outcome spaces

There are two main types of outcome spaces.

- Counting-compatible outcome spaces have a well-defined
    way of counting how often each point in the (encoded) input data is mapped to a
    particular outcome ``\\omega_i``. These outcome spaces use
    [`encode`](@ref) to discretize the input data. Examples are
    [`SymbolicPermutation`](@ref) (which encodes input data into ordinal patterns) or
    [`ValueHistogram`](@ref) (which discretizes points onto a regular grid).
    The table below lists which outcome spaces are counting compatible.
- Non-counting compatible outcome spaces have no well-defined way of counting explicitly
    how often each point in the input data is mapped to a particular outcome ``\\omega_i``.
    Instead, these outcome spaces returns a vector of pre-normalized "relative counts", one
    for each outcome ``\\omega_i``. Examples are [`WaveletOverlap`](@ref) or
    [`PowerSpectrum`](@ref).

Counting-compatible outcome spaces can be used with *any* [`ProbabilitiesEstimator`](@ref)
to convert counts into probability mass functions.
Non-counting-compatible outcome spaces can only be used with the maximum likelihood
([`MLE`](@ref)) probabilities estimator, which estimates probabilities precisely by the
relative frequency of each outcome (formally speaking, the [`MLE`](@ref) estimator also
requires counts, but for the sake of code consistency, we allow it to be used with
relative frequencies as well).

## Deducing the outcome space (from data)

Some outcome space models can deduce ``\\Omega`` without knowledge of the input, such as
[`SymbolicPermutation`](@ref). Other outcome spaces require knowledge of the input data
for concretely specifying ``\\Omega``, such as [`ValueHistogram`](@ref) with
[`RectangularBinning`](@ref). If `o` is some outcome space model and `x` some input data, then
[`outcome_space`](@ref)`(o, x)` returns the possible outcomes ``\\Omega``. To get the
cardinality of ``\\Omega``, use [`total_outcomes`](@ref).

## Implementation details

The element type of ``\\Omega`` varies between outcome space models, but it is guaranteed
to be _hashable_ and _sortable_. This allows for conveniently tracking the counts of a
specific event across experimental realizations, by using the outcome as a dictionary key
and the counts as the value for that key (or, alternatively, the key remains the outcome
and one has a vector of probabilities, one for each experimental realization).
"""
abstract type OutcomeSpace end

###########################################################################################
# Outcome space
###########################################################################################
"""
    outcome_space(o::OutcomeSpace, x) → Ω

Return a sorted container containing all _possible_ outcomes of `o` for input `x`.

For some estimators the concrete outcome space is known without knowledge of input `x`,
in which case the function dispatches to `outcome_space(est)`.
In general it is recommended to use the 2-argument version irrespectively of estimator.
"""
function outcome_space(est::OutcomeSpace)
    error(ErrorException("""
    `outcome_space(est)` not implemented for estimator $(typeof(est)).
    Try calling `outcome_space(est, input_data)`, and if you get the same error, open an issue.
    """))
end
outcome_space(o::OutcomeSpace, x) = outcome_space(o)

"""
    total_outcomes(o::OutcomeSpace, x)

Return the length (cardinality) of the outcome space ``\\Omega`` of `est`.

For some [`OutcomeSpace`](@ref), the cardinality is known without knowledge of input `x`,
in which case the function dispatches to `total_outcomes(est)`.
In general it is recommended to use the 2-argument version irrespectively of estimator.
"""
total_outcomes(o::OutcomeSpace, x) = length(outcome_space(o, x))
total_outcomes(o::OutcomeSpace) = length(outcome_space(o))

"""
    outcomes(o::OutcomeSpace, x)

Return all (unique) outcomes contained in `x` according to the given outcome space.
Equivalent to `probabilities_and_outcomes(o, x)[2]`, but for some estimators
it may be explicitly extended for better performance.
"""
function outcomes(o::OutcomeSpace, x)
    return last(counts_and_outcomes(o, x))
end

###########################################################################################
# Counts.
#
# The fundamental quantity used for probabilities estimation are *counts* of how often
# a certain outcome is observed in the input data. These counts are then translated into
# probability mass functions by dedicated `ProbabilitiesEstimator`s.
#
# For example, the most basic probabilities estimator is [`MLE`](@ref) - the maximum
# likelihood estimator - and it take the relative proportions of counts as the
# probabilities.
#
# If `counts_and_outcomes` and `allcounts_and_outcomes` are implemented for an
# `OutcomeSpace`, then the outcome space is automatically compatible with all
# `ProbabilitiesEstimator`s. For some outcome spaces, however, this is not possible,
# because counting is not defined over their outcome spaces (e.g. [`WaveletOverlap`](@ref)
#  use pre-normalized relative "frequencies", not counts, to estimate probabilities).
###########################################################################################
"""
    counts_and_outcomes(o::OutcomeSpace, x) → (cts::Vector{Int}, Ω::Vector)

Count how often each outcome `Ωᵢ ∈ Ω` appears in the (encoded) input data `x`, where
`Ω = outcome_space(o, x)`.

## Description

For [`OutcomeSpace`](@ref)s that uses [`encode`](@ref) to discretize, it is possible to
count how often each outcome ``\\omega_i \\in \\Omega``, where ``\\Omega`` is the
set of possible outcomes, is observed in the discretized/encoded input data.
Thus, we can assign to each outcome ``\\omega_i`` a count ``f(\\omega_i)``, such that
``\\sum_{i=1}^N f(\\omega_i) = N``, where ``N`` is the number of observations in the
(encoded) input data.
`counts_and_outcomes` returns the counts ``f(\\omega_i)_{obs}``
and outcomes only for the *observed* outcomes ``\\Omega_i^{obs}`` (those outcomes
that actually appear in the input data). If you need the counts for
*unobserved* outcomes as well, use [`allcounts_and_outcomes`](@ref).

Returns the `cts` and `Ω` as a tuple where `length(cts) == length(Ω)`.
"""
function counts_and_outcomes(o::OutcomeSpace, x)
    throw(ArgumentError("`counts_and_outcomes` not implemented for estimator $(typeof(o))."))
end

"""
    allcounts_and_outcomes(o::OutcomeSpace, x::Array_or_SSSet) → (cts::Vector{Int}, Ω::Vector)

Like [`counts_and_outcomes`](@ref), but ensures that *all* outcomes `Ωᵢ ∈ Ω`,
where `Ω = outcome_space(o, x)`), are included.

Returns the `cts` and `Ω` as a tuple where `length(cts) == length(Ω)`.
"""
function allcounts_and_outcomes(o::OutcomeSpace, x::Array_or_SSSet)
    freqs, outs = counts_and_outcomes(o, x)
    ospace = vec(outcome_space(o, x))
    # We first utilize that the outcome space is sorted and sort probabilities
    # accordingly (just in case we have an estimator that is not sorted)
    s = sortperm(outs)
    sort!(outs)
    fs = freqs[s]
    # we now iterate over possible outcomes;
    # if they exist in the observed outcomes, we push their corresponding frequency
    # into the frequencies vector. If not, we push 0 into the frequencies vector!
    allfreqs = eltype(fs)[]
    observed_index = 1 # index of observed outcomes
    for j in eachindex(ospace) # we made outcome space a vector on purpose
        ω = ospace[j]
        ωobs = outs[observed_index]
        if ω ≠ ωobs
            push!(allfreqs, 0)
        else
            push!(allfreqs, fs[observed_index])
            observed_index += 1
        end
        # Check whether we have exhausted observed outcomes
        if observed_index > length(outs)
            remaining_0s = length(ospace) - j
            append!(allfreqs, zeros(Int, remaining_0s))
            break
        end
    end
    return allfreqs, ospace
end

"""
    allcounts(o::OutcomeSpace, x::Array_or_SSSet) → cts::Vector{Int}

Like [`allcounts_and_outcomes`](@ref), but only returns the counts.
"""
allcounts(o::OutcomeSpace, x::Array_or_SSSet) = first(allcounts_and_outcomes(o, x))

"""
    counts(o::OutcomeSpace, x) → cts::Vector{Int}

Like [`counts_and_outcomes`](@ref), but only returns the counts.
"""
function counts(o::OutcomeSpace, x)
    return first(counts_and_outcomes(o, x))
end
