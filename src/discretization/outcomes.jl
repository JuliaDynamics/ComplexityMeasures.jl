export discretize, Discretization

"""
    Discretization

The supertype for all discretization schemes, i.e. ways of coarse-graining input data,
for example when mapping data to some outcome space when estimating [`probabilities`](@ref).
"""
abstract type Discretization end

"""
    outcomes(x, scheme::Discretization) → Vector{Int}
    outcomes!(s, x, scheme::Discretization) → Vector{Int}

Map each`xᵢ ∈ x` to a distinct [outcome](@ref terminology) according to the
discretization `scheme`.

Optionally, write outcomes into the pre-allocated symbol vector `s` if the `scheme`
allows for it. For usage examples, see individual discretization scheme docstrings.

The following discretization schemes are currently implemented:
- [`OrdinalMapping`](@ref).
- [`GaussianMapping`](@ref).
- [`RectangularBinMapping`](@ref).

Used internally by the various [`ProbabilitiesEstimator`](@ref)s to define
outcome spaces over which to compute probabilities.
"""
function outcomes(x, ::Discretization) end

# The internal structure of different discretization schemes may be different, so use
# `total_outcomes` to have a consistent way of getting the total number of possible states.
# Thus, the default behaviour is to throw an ArgumentError when computing some normalized
# quantity depending on `total_outcomes` of the discretization scheme.
total_outcomes(s::S) where S <: Discretization =
    throw(ArgumentError("`total_outcomes` not defined for $S."))

include("utils.jl")
include("gaussian_mapping.jl")
include("ordinal_mapping.jl")
