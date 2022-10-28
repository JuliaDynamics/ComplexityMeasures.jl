export discretize, Encoding

"""
    Encoding

The supertype for all encoding schemes, i.e. ways of coarse-graining input data,
for example when mapping data to some outcome space when estimating [`probabilities`](@ref).
"""
abstract type Encoding end

"""
    outcomes(x, scheme::Encoding) → Vector{Int}
    outcomes!(s, x, scheme::Encoding) → Vector{Int}

Map each`xᵢ ∈ x` to a distinct [outcome](@ref terminology) according to the
encoding `scheme`.

Optionally, write outcomes into the pre-allocated symbol vector `s` if the `scheme`
allows for it. For usage examples, see individual encoding scheme docstrings.

The following encoding schemes are currently implemented:
- [`OrdinalMapping`](@ref).
- [`GaussianMapping`](@ref).
- [`RectangularBinMapping`](@ref).

Used internally by the various [`ProbabilitiesEstimator`](@ref)s to define
outcome spaces over which to compute probabilities.
"""
function outcomes(x, ::Encoding) end

# The internal structure of different encoding schemes may be different, so use
# `total_outcomes` to have a consistent way of getting the total number of possible states.
# Thus, the default behaviour is to throw an ArgumentError when computing some normalized
# quantity depending on `total_outcomes` of the encoding scheme.
total_outcomes(s::S) where S <: Encoding =
    throw(ArgumentError("`total_outcomes` not defined for $S."))

include("utils.jl")
include("gaussian_mapping.jl")
include("ordinal_mapping.jl")
