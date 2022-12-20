"""
A Julia package that provides estimators for probabilities and entropies for nonlinear
dynamics, nonlinear timeseries analysis, and complex systems. It can be used as a
standalone package, or as part of several projects in the JuliaDynamics organization,
such as [DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystems.jl/dev/)
or [CausalityTools.jl](https://juliadynamics.github.io/CausalityTools.jl/dev/).

To install it, run `import Pkg; Pkg.add("Entropies")`.
"""
module Entropies

using StateSpaceSets
export Dataset, SVector
using DelayEmbeddings: embed, genembed

const Array_or_Dataset = Union{<:AbstractArray{<:Real}, <:AbstractDataset}
const Vector_or_Dataset = Union{<:AbstractVector{<:Real}, <:AbstractDataset}

# Core API types and functions
include("probabilities.jl")
include("entropy.jl")
include("encodings.jl")
include("complexity.jl")
include("multiscale.jl")

# Library implementations (files include other files)
include("probabilities_estimators/probabilities_estimators.jl")
include("entropies/entropies.jl")
include("encoding/all_encodings.jl")
include("complexity/complexity_measures.jl") # relies on encodings, so include after
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
