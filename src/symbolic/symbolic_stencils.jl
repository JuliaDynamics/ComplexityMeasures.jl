"""
    SpatiotemporalPermutation(stencil, periodic = true)
A symbolic, permutation-based probabilities/entropy estimator for higher-dimensional
data, typically representing spatiotemporal systems. The data are higher-dimensional
arrays, such as 2D [^Ribeiro2012] or 3D [^Schlemmer2018] `Matrix`.
This approach is also known as _Spatiotemporal Permutation Entropy_.

A _stencil_ defines what local area around each pixel to
consider, and compute the ordinal pattern within the stencil. Stencils are given as
vectors of `CartesianIndex` which encode the _offsets_ of the pixes to include in the
stencil, with respect to the current pixel. For example
```julia
stencil = CartesianIndex.([(0,1), (0,1), (1,1)])
est = StencilSymbolicPermutation(stencil)
```
Here the stencil creates a 2x2 square extending to the bottom and right of the pixel.
Notice that offset `0` (meaning the pixel itself) is alwaus inclided automatically.
The length of the stencil decides the order of the permutation entropy, and the ordering
within the stencil lifts any ambiguity regarding the order that pixels are compared to.
The pixel without any offset is always first in the order.

After having defined `est`, one calculates the permutation entropy of ordinal patterns
by calling [`genentropy`](@ref) with `est`, and with the data that will be a matrix.
To apply this to timeseries of spatial patterns, simply loop over the call, e.g.:
```julia
data = [rand(50,50) for _ in 1:50]
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
struct SpatiotemporalPermutation{D,P} <: ProbabilityEstimator
    stencil::Vector{CartesianIndex{D}}
end
function SpatiotemporalPermutation(stencil::Vector{CartesianIndex{D}}, p::Bool) where {D}
    SpatiotemporalPermutation{D, p}(stencil)
end
