module Entropies

# Use the README as the module docs
@doc let
  path = joinpath(dirname(@__DIR__), "README.md")
  include_dependency(path)
  read(path, String)
end Entropies

using Reexport
@reexport using StateSpaceSets
using DelayEmbeddings: embed

const Array_or_Dataset = Union{<:AbstractArray{<:Real}, <:AbstractDataset}
const Vector_or_Dataset = Union{<:AbstractVector{<:Real}, <:AbstractDataset}

# Core API types and functions
include("probabilities.jl")
include("entropy.jl")
include("encodings.jl")
include("complexity.jl")
include("multiscale.jl")

# Library implementations (files include other files)
include("encoding/all_encodings.jl") # other structs depend on these
include("probabilities_estimators/probabilities_estimators.jl")
include("entropies_definitions/entropies_definitions.jl")
include("complexity/complexity_measures.jl")
include("deprecations.jl")


# Update messages:
using Scratch
display_update = true
version_number = "2.0.0"
update_name = "update_v$(version_number)"
update_message = """
\nUpdate message: Entropies v$(version_number)\n
- An overall overhaul of the documentation and API of Entropies.jl has been performed.
- A huge amount of new content has been added, which is best seen by visiting the
  the online documentation. Some examples are Tsallis entropy and spatial permutation
  entropy, and much more.
- In summary, all entropies and normalized entropies are computing using the
  `entropy` and `entropy_normalized` functions, which dispatch on entropy types such
  as `Renyi()`, `Shannon()` or `Tsallis()`.
- New constructors for discretizing: `OrdinalPatternEncoding, GaussianCDFEncoding`.
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
