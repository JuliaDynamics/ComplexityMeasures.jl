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
include("core/outcome.jl")
include("core/counts.jl")
include("core/probabilities.jl")
include("core/print_counts_probs.jl") # pretty printing
include("core/information_measures.jl")
include("core/information_functions.jl")
include("core/encodings.jl")
include("core/complexity.jl")
include("multiscale.jl")
include("convenience.jl")
include("core/pretty_printing.jl")

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

end
