"""
    SpatiotemporalPermutation(stencil, periodic = true)
A symbolic, permutation-based probabilities/entropy estimator for higher-dimensional
data, typically representing spatiotemporal systems. The data are higher-dimensional
arrays, such as 2D [^Ribeiro2012] or 3D [^Schlemmer2018].
This approach is also known as _Spatiotemporal Permutation Entropy_.

A _stencil_ defines what local area around each pixel to
consider, and compute the ordinal pattern within the stencil. Stencils are given as
vectors of `CartesianIndex` which encode the _offsets_ of the pixes to include in the
stencil, with respect to the current pixel. For example
```julia
stencil = CartesianIndex.([(0,1), (0,1), (1,1)])
est = SpatiotemporalPermutation(stencil)
```
Here the stencil creates a 2x2 square extending to the bottom and right of the pixel.
Notice that no offset (meaning the pixel itself) is always included automatically.
The length of the stencil decides the order of the permutation entropy, and the ordering
within the stencil dictates the order that pixels are compared with.
The pixel without any offset is always first in the order.

After having defined `est`, one calculates the permutation entropy of ordinal patterns
by calling [`genentropy`](@ref) with `est`, and with the array data.
To apply this to timeseries of spatial data, simply loop over the call, e.g.:
```julia
data = [rand(50, 50) for _ in 1:50]
entropies = [genentropy(x, est) for x in data]
```

The argument `periodic` decides whether the stencil should wrap around at the end of the
array. If `periodic = false`, pixels whose stencil exceeds the array bounds are skipped.

[^Ribeiro2012]:
    Ribeiro et al. (2012). Complexity-entropy causality plane as a complexity measure
    for two-dimensional patterns. https://doi.org/10.1371/journal.pone.0040689

[^Schlemmer2018]:
    Schlemmer et al. (2012). Spatiotemporal Permutation Entropy as a Measure for
    Complexity of Cardiac Arrhythmia. https://doi.org/10.3389/fphy.2018.00039
"""
struct SpatiotemporalPermutation{D,P} <: ProbabilitiesEstimator
    stencil::Vector{CartesianIndex{D}}
    viewer::Vector{CartesianIndex{D}}
end
function SpatiotemporalPermutation(
        stencil::Vector{CartesianIndex{D}}, p::Bool = true
    ) where {D}
    # Ensure that no offset is part of the stencil
    stencil = pushfirst!(copy(stencil), CartesianIndex{D}(zeros(Int, D)...))
    SpatiotemporalPermutation{D, p}(stencil, copy(stencil))
end

# This source code is a modification of the code of Agents.jl that finds neighbors
# in grid-like spaces. It's the code of `nearby_positions` in `grid_general.jl`.
function pixels_in_stencil(array, pixel, spatperm::SpatiotemporalPermutation{D,false}) where {D}
    # TODO: Not clear yet how out of bounds is handled in this case.
    @inbounds for i in eachindex(spatperm.stencil)
        statperm.viewer[i] = spatperm.stencil[i] + pixel
    end
    return statperm.viewer
    # space_size = size(array)
    # positions_iterator = (n + pixel for n in spatperm.stencil)
    # return Base.Iterators.filter(
    #     pos -> checkbounds(Bool, array, pos...), positions_iterator
    # )
end

function pixels_in_stencil(array, pixel, spatperm::SpatiotemporalPermutation{D,true}) where {D}
    space_size = size(array)
    @inbounds for i in eachindex(spatperm.stencil)
        spatperm.viewer[i] = CartesianIndex{D}(mod1.(Tuple(spatperm.stencil[i] + pixel), space_size))
    end
    return spatperm.viewer
end

function Entropies.probabilities(x, est::SpatiotemporalPermutation) where {m, T}
    s = zeros(Int, length(x))
    probabilities!(s, x, est)
end

function Entropies.probabilities!(s::AbstractVector{Int}, x, est::SpatiotemporalPermutation)
    m = length(est.stencil)
    for (i, pixel) in enumerate(CartesianIndices(x))
        pixels = pixels_in_stencil(x, pixel, est)
        s[i] = Entropies.encode_motif(view(x, pixels), m)
    end
    return probabilities(s)
end

# %%
stencil = CartesianIndex.([(0,1), (0,1), (1,1)])
est = SpatiotemporalPermutation(stencil)
x = rand(50, 50)
p = probabilities(x, est)
