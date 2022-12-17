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
"""
abstract type Encoding end

"""
    encode(χ, e::Encoding) -> i::Int
Encoding an element `χ ∈ x` of input data `x` (those given to [`probabilities`](@ref))
using encoding `e`. The special value of `-1` is reserved as a return value for
inappropriate elements `χ` that cannot be encoded according to `e`.
"""
function encode end

"""
    decode(i::Int, e::Encoding) -> ω
Decode an encoded element `i::Int` into the outcome it corresponds to `ω ∈ Ω`.
`Ω` is the [`outcome_space`](@ref) of a probabilities estimator that uses encoding `e`.
"""
function decode end