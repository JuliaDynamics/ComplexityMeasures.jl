export symbolize

abstract type SymbolizationScheme end

"""
    symbolize(x, scheme::SymbolizationScheme) → Vector{Int}
    symbolize!(s, x, scheme::SymbolizationScheme) → Vector{Int}

Symbolize `x` using the provided symbolization `scheme`, optionally writing symbols into the
pre-allocated symbol vector `s`. For usage examples, see individual symbolization scheme
docstrings.

See also: [`OrdinalPattern`](@ref), [`GaussianSymbolization`](@ref).
"""
function symbolize end

include("utils.jl")
include("GaussianSymbolization.jl")
include("OrdinalPattern.jl")
