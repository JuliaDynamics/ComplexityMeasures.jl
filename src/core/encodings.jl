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
    decode(c::Encoding, i::Integer) -> ω

Decode an encoded element `i` into the outcome `ω ∈ Ω` it corresponds to.

`Ω` is the [`outcome_space`](@ref) of a probabilities estimator that uses encoding `c`.
"""
function decode end
