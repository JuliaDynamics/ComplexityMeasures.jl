export OutcomeSpace
export outcome_space
export total_outcomes
export outcomes

"""
    OutcomeSpace

The supertype for all outcome space implementation.

## Description

In ComplexityMeasures.jl, an outcome space defines a set of possible outcomes
``\\Omega = \\{\\omega_1, \\omega_2, \\ldots, \\omega_L \\}`` (some form of
discretization). In the literature, the outcome space is often called an "alphabet",
while each outcome is called a "symbol" or an "event".

An outcome space also defines a set of rules for mapping input data to
to each outcome ``\\omega_i`` (i.e. [encoding](@ref encodings)/discretizing).
Some [`OutcomeSpace`](@ref)s first apply a transformation, e.g. a delay embedding, to
the data before discretizing/encoding, while other [`OutcomeSpace`](@ref)s
discretize/encode the data directly.

## Implementations

| Outcome space                           | Principle                    | Input data                | Counting-compatible |
| :-------------------------------------- | :--------------------------- | :------------------------ | :------------------ |
| [`CountOccurrences`](@ref)              | Count of unique elements     | `Any`                     | ✔                  |
| [`ValueHistogram`](@ref)                | Binning (histogram)          | `Vector`, `StateSpaceSet` | ✔                  |
| [`TransferOperator`](@ref)              | Binning (transfer operator)  | `Vector`, `StateSpaceSet` | ✖                  |
| [`NaiveKernel`](@ref)                   | Kernel density estimation    | `StateSpaceSet`           | ✖                  |
| [`OrdinalPatterns`](@ref)               | Ordinal patterns             | `Vector`, `StateSpaceSet` | ✔                  |
| [`WeightedOrdinalPatterns`](@ref)       | Ordinal patterns             | `Vector`, `StateSpaceSet` | ✖                  |
| [`AmplitudeAwareOrdinalPatterns`](@ref) | Ordinal patterns             | `Vector`, `StateSpaceSet` | ✖                  |
| [`SpatialOrdinalPatterns`](@ref)        | Ordinal patterns in space    | `Array`                   | ✔                  |
| [`Dispersion`](@ref)                    | Dispersion patterns          | `Vector`                  | ✔                  |
| [`SpatialDispersion`](@ref)             | Dispersion patterns in space | `Array`                   | ✔                  |
| [`Diversity`](@ref)                     | Cosine similarity            | `Vector`                  | ✔                  |
| [`WaveletOverlap`](@ref)                | Wavelet transform            | `Vector`                  | ✖                  |
| [`PowerSpectrum`](@ref)                 | Fourier transform            | `Vector`                  | ✖                  |

In the column "input data" it is assumed that the `eltype` of the input is `<: Real`.

## Usage

Outcome spaces are used as input to

- [`probabilities`](@ref)/[`allprobabilities`](@ref) for computing probability
    mass functions.
- [`outcome_space`](@ref), which returns the elements of the outcome space.
- [`total_outcomes`](@ref), which returns the cardinality of the outcome space.
- [`counts`](@ref)/[`allcounts`](@ref), for obtaining raw counts instead
    of probabilities (only for counting-compatible outcome spaces).

## Counting-compatible vs. non-counting compatible outcome spaces

There are two main types of outcome spaces.

- Counting-compatible outcome spaces have a well-defined
    way of counting how often each point in the (encoded) input data is mapped to a
    particular outcome ``\\omega_i``. These outcome spaces use
    [`encode`](@ref) to discretize the input data. Examples are
    [`OrdinalPatterns`](@ref) (which encodes input data into ordinal patterns) or
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
([`RelativeAmount`](@ref)) probabilities estimator, which estimates probabilities precisely by the
relative frequency of each outcome (formally speaking, the [`RelativeAmount`](@ref) estimator also
requires counts, but for the sake of code consistency, we allow it to be used with
relative frequencies as well).

The function [`is_counting_based`](@ref) can be used to check whether an outcome space
is based on counting.

## Deducing the outcome space (from data)

Some outcome space models can deduce ``\\Omega`` without knowledge of the input, such as
[`OrdinalPatterns`](@ref). Other outcome spaces require knowledge of the input data
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

# Use this subtype instead for outcome spaces that support counting
abstract type CountBasedOutcomeSpace <: OutcomeSpace end

###########################################################################################
# Outcome space
###########################################################################################
"""
    outcome_space(o::OutcomeSpace, x) → Ω

Return a sorted container containing all _possible_ outcomes of `o` for input `x`.

For some estimators the concrete outcome space is known without knowledge of input `x`,
in which case the function dispatches to `outcome_space(o)`.
In general it is recommended to use the 2-argument version irrespectively of estimator.
"""
function outcome_space(o::OutcomeSpace)
    error(ErrorException("""
    `outcome_space(o)` not implemented for outcome space $(typeof(o)).
    Try calling `outcome_space(o, input_data)`, and if you get the same error, open an issue.
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

Return all (unique) outcomes that appears in the (encoded) input data `x`,
according to the given [`OutcomeSpace`](@ref).
Equivalent to `probabilities_and_outcomes(o, x)[2]`, but for some estimators
it may be explicitly extended for better performance.
"""
function outcomes(o::OutcomeSpace, x)
    if is_counting_based(o)
        return last(counts_and_outcomes(o, x))
    else
        return last(probabilities_and_outcomes(o, x))
    end
end
