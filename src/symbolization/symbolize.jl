export symbolize

abstract type SymbolizationScheme end

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
