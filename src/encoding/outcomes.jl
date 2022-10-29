export discretize, Encoding

"""
    Encoding

The supertype for all encoding schemes, i.e. ways of encoding input data onto some
discrete set of possibilities/[outcomes](@ref). Implemented encoding schemes are

- [`OrdinalPatternEncoding`](@ref).
- [`GaussianCDFEncoding`](@ref).
- [`RectangularBinEncoding`](@ref).

Used internally by the various [`ProbabilitiesEstimator`](@ref)s to map input data onto
outcome spaces, over which probabilities are computed.
"""
abstract type Encoding end

"""
    outcomes(x, scheme::Encoding) → Vector{Int}
    outcomes!(s, x, scheme::Encoding) → Vector{Int}

Map each`xᵢ ∈ x` to a distinct outcome according to the encoding `scheme`.

Optionally, write outcomes into the pre-allocated symbol vector `s` if the `scheme`
allows for it. For usage examples, see individual encoding scheme docstrings.

See also: [`RectangularBinEncoding`](@ref), [`GaussianCDFEncoding`](@ref),
[`OrdinalPatternEncoding`](@ref).
"""
function outcomes(x::X, ::Encoding) where X
    throw(ArgumentError("`outcomes` not defined for input data of type $(X)."))
end

# The internal structure of different encoding schemes may be different, so use
# `total_outcomes` to have a consistent way of getting the total number of possible states.
# Thus, the default behaviour is to throw an ArgumentError when computing some normalized
# quantity depending on `total_outcomes` of the encoding scheme.
total_outcomes(s::S) where S <: Encoding =
    throw(ArgumentError("`total_outcomes` not defined for $S."))

include("utils.jl")
include("gaussian_cdf.jl")
include("ordinal_pattern.jl")
