export symbolize, SymbolizationScheme

"""
An abstract type for symbolization schemes.
"""
abstract type SymbolizationScheme end

# The internal structure of different symbolization schemes may be different, so use
# `alphabet_length` to have a consistent way of getting the total number of possible states.
# Thus, the default behaviour is to throw an ArgumentError when computing some normalized
# quantity depending on `alphabet_length` of the symbolization scheme.
alphabet_length(s::S) where S <: SymbolizationScheme =
    throw(ArgumentError("`alphabet_length` not defined for $S."))

"""
    symbolize(x, scheme::SymbolizationScheme) → Vector{Int}
    symbolize!(s, x, scheme::SymbolizationScheme) → Vector{Int}

Symbolize `x` using the provided symbolization `scheme`, optionally writing symbols into the
pre-allocated symbol vector `s` if the `scheme` allows for it.
For usage examples, see individual symbolization scheme docstrings.

The following symbolization schemes are currently implemented:
- [`OrdinalPattern`](@ref).
- [`GaussianSymbolization`](@ref).
- [`RectangularBinEncoder`](@ref).
"""
function symbolize end

include("utils.jl")
include("GaussianSymbolization.jl")
include("OrdinalPattern.jl")
