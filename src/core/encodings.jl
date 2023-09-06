export Encoding, encode, decode

"""
    Encoding

The supertype for all encoding schemes. Encodings always encode elements of
input data into the positive integers. The encoding API is defined by the
functions [`encode`](@ref) and [`decode`](@ref).
Some probability estimators utilize encodings internally.

Current available encodings are:

- [`OrdinalPatternEncoding`](@ref).
- [`GaussianCDFEncoding`](@ref).
- [`RectangularBinEncoding`](@ref).
- [`RelativeMeanEncoding`](@ref).
- [`RelativeFirstDifferenceEncoding`](@ref).
- [`CombinationEncoding`](@ref), which can combine any of the above encodings.
"""
abstract type Encoding end

"""
    encode(c::Encoding, χ) -> i::Int

Encode an element `χ ∈ x` of input data `x` (those given to [`probabilities`](@ref))
using encoding `c`.

The special value of `-1` is reserved as a return value for
inappropriate elements `χ` that cannot be encoded according to `c`.
"""
function encode end

"""
    decode(c::Encoding, i::Int) -> ω

Decode an encoded element `i` into the outcome `ω ∈ Ω` it corresponds to.

`Ω` is the [`outcome_space`](@ref) of a probabilities estimator that uses encoding `c`.
"""
function decode end

export symbolize

# `symbolize` is the function that actually performs the transformation of the
# input data to elements of the (encoded) outcome space. This method is used both internally
# here, where relevant, but is most important upstream, where it is used to ensure
# that multivariate time series data is always encoded to integers.
"""

    symbolize(o::OutcomeSpace, x::Vector) → s::Vector{Int}
    symbolize(o::OutcomeSpace, x::AbstractStateSpaceSet{D}) → s::NTuple{D, Vector{Int}

Symbolize `x` according to the outcome space `o`.

## Description

The reason this function exists is that we don't always want to [`encode`](@ref) the
entire input `x` at once. Sometimes, it is desirable to first apply some transformation to
`x` first, then apply [`encoding`](@ref)s in a point-wise manner in the transformed space.
The [`OutcomeSpace`](@ref) dictates this transformation. This is useful for encoding
time series data.

The length of the returned `s` depends on the [`OutcomeSpace`](@ref). Some outcome
spaces preserve the input data length (e.g. [`CountOccurrences`](@ref)), while
some outcome spaces (e.g. [`OrdinalPatterns`](@ref)) do e.g. delay embeddings before
encoding, so that `length(s) < length(x)`.

## Returns

If `x` is a `Vector`, then a `Vector{<:Integer}` is returned. If `x` is a
`StateSpaceSet{D}`, then symbolization is done column-wise and an
`NTuple{D, Vector{<:Integer}}` is returned, where `D = dimension(x)`.

# Concrete implementations

    symbolize(o::CountOccurrences, x::VectorOrStateSpaceSet)
    symbolize(o::OrdinalPatterns, x::VectorOrStateSpaceSet)
    symbolize(o::Dispersion, x::VectorOrStateSpaceSet)

These are listed for convenience.
"""
function symbolize end
