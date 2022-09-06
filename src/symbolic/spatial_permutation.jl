export SpatialSymbolicPermutation

"""
    SpatialSymbolicPermutation(stencil, x, periodic = true)
A symbolic, permutation-based probabilities/entropy estimator for spatiotemporal systems.
The data are a high-dimensional array `x`, such as 2D [^Ribeiro2012] or 3D [^Schlemmer2018].
This approach is also known as _Spatiotemporal Permutation Entropy_.
`x` is given because we need to know its size for optimization and bound checking.

A _stencil_ defines what local area around each pixel to
consider, and compute the ordinal pattern within the stencil. Stencils are given as
vectors of `CartesianIndex` which encode the _offsets_ of the pixes to include in the
stencil, with respect to the current pixel. For example
```julia
data = [rand(50, 50) for _ in 1:50]
x = data[1] # first "time slice" of a spatial system evolution
stencil = CartesianIndex.([(0,1), (1,1), (1,0)])
est = SpatialSymbolicPermutation(stencil, x)
```
Here the stencil creates a 2x2 square extending to the bottom and right of the pixel
(directions here correspond to the way Julia prints matrices by default).
Notice that no offset (meaning the pixel itself) is always included automatically.
The length of the stencil decides the order of the permutation entropy, and the ordering
within the stencil dictates the order that pixels are compared with.
The pixel without any offset is always first in the order.

After having defined `est`, one calculates the permutation entropy of ordinal patterns
by calling [`entropy_renyi`](@ref) with `est`, and with the array data.
To apply this to timeseries of spatial data, simply loop over the call, e.g.:
```julia
entropy = entropy_renyi(x, est)
entropy_vs_time = entropy_renyi.(data, est) # broadcasting with `.`
```

The argument `periodic` decides whether the stencil should wrap around at the end of the
array. If `periodic = false`, pixels whose stencil exceeds the array bounds are skipped.

[^Ribeiro2012]:
    Ribeiro et al. (2012). Complexity-entropy causality plane as a complexity measure
    for two-dimensional patterns. https://doi.org/10.1371/journal.pone.0040689

[^Schlemmer2018]:
    Schlemmer et al. (2018). Spatiotemporal Permutation Entropy as a Measure for
    Complexity of Cardiac Arrhythmia. https://doi.org/10.3389/fphy.2018.00039
"""
struct SpatialSymbolicPermutation{D,P,V} <: ProbabilitiesEstimator
    stencil::Vector{CartesianIndex{D}}
    viewer::Vector{CartesianIndex{D}}
    arraysize::Dims{D}
    valid::V
end
function SpatialSymbolicPermutation(
        stencil::Vector{CartesianIndex{D}}, x::AbstractArray, p::Bool = true
    ) where {D}
    # Ensure that no offset is part of the stencil
    stencil = pushfirst!(copy(stencil), CartesianIndex{D}(zeros(Int, D)...))
    arraysize = size(x)
    @assert length(arraysize) == D "Indices and input array must match dimensionality!"
    # Store valid indices for later iteration
    if p
        valid = CartesianIndices(x)
    else
        # collect maximum offsets in each dimension for limiting ranges
        maxoffsets = [maximum(s[i] for s in stencil) for i in 1:D]
        # Safety check
        minoffsets = [min(0, minimum(s[i] for s in stencil)) for i in 1:D]
        ranges = Iterators.product(
            [(1-minoffsets[i]):(arraysize[i]-maxoffsets[i]) for i in 1:D]...
        )
        valid = Base.Generator(idxs -> CartesianIndex{D}(idxs), ranges)
    end
    SpatialSymbolicPermutation{D, p, typeof(valid)}(stencil, copy(stencil), arraysize, valid)
end

# This source code is a modification of the code of Agents.jl that finds neighbors
# in grid-like spaces. It's the code of `nearby_positions` in `grid_general.jl`.
function pixels_in_stencil(pixel, spatperm::SpatialSymbolicPermutation{D,false}) where {D}
    @inbounds for i in eachindex(spatperm.stencil)
        spatperm.viewer[i] = spatperm.stencil[i] + pixel
    end
    return spatperm.viewer
end

function pixels_in_stencil(pixel, spatperm::SpatialSymbolicPermutation{D,true}) where {D}
    @inbounds for i in eachindex(spatperm.stencil)
        # It's annoying that we have to change to tuple and then to CartesianIndex
        # because iteration over cartesian indices is not allowed. But oh well.
        spatperm.viewer[i] = CartesianIndex{D}(
            mod1.(Tuple(spatperm.stencil[i] + pixel), spatperm.arraysize)
        )
    end
    return spatperm.viewer
end

function Entropies.probabilities(x, est::SpatialSymbolicPermutation)
    # TODO: This can be literally a call to `symbolize` and then
    # calling probabilities on it. Should do once the `symbolize` refactoring is done.
    s = zeros(Int, length(est.valid))
    probabilities!(s, x, est)
end

function Entropies.probabilities!(s, x, est::SpatialSymbolicPermutation)
    m = length(est.stencil)
    for (i, pixel) in enumerate(est.valid)
        pixels = pixels_in_stencil(pixel, est)
        s[i] = Entropies.encode_motif(view(x, pixels), m)
    end
    return probabilities(s)
end

# Pretty printing
function Base.show(io::IO, est::SpatialSymbolicPermutation{D}) where {D}
    print(io, "Spatial permutation estimator for $D-dimensional data. Stencil:")
    print(io, "\n")
    show(io, MIME"text/plain"(), est.stencil)
end
