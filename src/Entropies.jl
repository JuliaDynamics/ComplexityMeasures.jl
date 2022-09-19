"""
A Julia package that provides estimators for probabilities, entropies,
and complexity measures for timeseries, nonlinear dynamics and complex systems.
It can be used as a standalone package, or as part of several projects in the JuliaDynamics organization,
such as [DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystems.jl/dev/)
or [CausalityTools.jl](https://juliadynamics.github.io/CausalityTools.jl/dev/).
"""
module Entropies

using DelayEmbeddings
using DelayEmbeddings: AbstractDataset, Dataset, dimension
export AbstractDataset, Dataset
const Array_or_Dataset = Union{<:AbstractArray, <:AbstractDataset}

include("symbolization/symbolize.jl")
include("probabilities.jl")
include("probabilities_estimators/probabilities_estimators.jl")
include("entropies/entropies.jl")
include("deprecations.jl")


# Update messages:
using Scratch
display_update = true
version_number = "1.3.0"
update_name = "update_v$(version_number)"
update_message = """
\nUpdate message: Entropies v$(version_number)\n
- An overall overhaul of the documentation and API of Entropies.jl has been performed.
  Now `genentropy` is `entropy_renyi`. The documentation further clarifies
  the difference between calculating probabilities and entropies.
- A huge amount of new content has been added to Entropies, which is best seen
  by visiting the changelog or the new documentation.
  E.g., Tsallis entropy, spatial permutation entropy, dispersion entropy, and many more.
- New exported API function: `symbolize`.
- New constructors for symbolizing: `OrdinalPattern, GaussianSymbolization`.
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
