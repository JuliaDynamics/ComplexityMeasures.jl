module ComplexityMeasures

# Use the README as the module docs
@doc let
  path = joinpath(dirname(@__DIR__), "README.md")
  include_dependency(path)
  read(path, String)
end ComplexityMeasures

using Reexport
@reexport using StateSpaceSets
using DelayEmbeddings: embed

const Array_or_SSSet = Union{<:AbstractArray{<:Real}, <:AbstractStateSpaceSet}
const Vector_or_SSSet = Union{<:AbstractVector{<:Real}, <:AbstractStateSpaceSet}

# Core API types and functions
include("core/outcome_spaces.jl")
include("core/probabilities.jl")
include("core/information_measures.jl")
include("core/information_functions.jl")
include("core/encodings.jl")
include("core/complexity.jl")
include("multiscale.jl")
include("convenience.jl")

# Library implementations (files include other files)
include("encoding_implementations/encoding_implementations.jl")
include("outcome_spaces/outcome_spaces.jl")
include("probabilities_estimators/probabilities_estimators.jl")
include("information_measure_definitions/information_measure_definitions.jl")
include("differential_info_estimators/differential_info_estimators.jl")
include("discrete_info_estimators/discrete_info_estimators.jl")
include("complexity_measures/complexity_measures.jl")

# deprecations (must be after all declarations)
include("deprecations.jl")

# Update messages:
using Scratch
display_update = false
version_number = "2.0"
update_name = "update_v$(version_number)"
update_message = """
Update message: ComplexityMeasures.jl v$(version_number)!

Entropies.jl has been completely overhauled,
and has been renamed to ComplexityMeasures.jl.

Along with the overhaul comes a massive amount of new features, an entirely new API,
extendable and educative code, dedicated documentation pages, and more!
At the moment we estimate we offer at least 90 unique quantities characterizing
complexity in the context of nonlinear dynamics and complex systems.
We believe it is best to learn all of this by visiting the online documentation!

We tried our best to keep pre-2.0 functions working and throw deprecation warnings.
If we missed code that should be working, please let us know by opening an issue.

For example, `geninformation(x::Array_or_SSSet, ε::Real; q, base)` is deprecated
in favor of `information(Renyi(q, base), ValueHistogram(ε), x)`.
"""

if display_update
    # Get scratch space for this package
    versions_dir = @get_scratch!("versions")
    if !isfile(joinpath(versions_dir, update_name))
        printstyled(
            stdout,
            update_message;
            color = :light_magenta,
        )
        touch(joinpath(versions_dir, update_name))
    end
end

end
