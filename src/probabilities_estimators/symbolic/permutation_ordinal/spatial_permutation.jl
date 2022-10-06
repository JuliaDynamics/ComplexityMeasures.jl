export SpatialSymbolicPermutation

"""
    SpatialSymbolicPermutation(stencil, x; periodic = true)

A symbolic, permutation-based probabilities/entropy estimator for spatiotemporal systems.

The input data `x` are high-dimensional arrays, for example 2D arrays [^Ribeiro2012] or 3D arrays
[^Schlemmer2018]. This approach is also known as _spatiotemporal permutation entropy_.
`x` is given because we need to know its size for optimization and bound checking.

A _stencil_ defines what local area (which points) around each pixel to
consider, and compute ordinal patterns from.
The number of included points in a stencil (`m`) determines the length of the vectors
to be symbolized, i.e. there are `m!` possible ordinal patterns around each pixel.

Example usage:
```julia
data = [rand(50, 50) for _ in 1:50]
x = data[1] # first "time slice" of a spatial system evolution
stencil = ...
est = SpatialSymbolicPermutation(stencil, x)
```

Stencils are passed in one of the following three ways:

1. As vectors of `CartesianIndex` which encode the pixels to include in the
    stencil, with respect to the current pixel, or integer arrays of the same dimensionality
    as the data. For example

    ```julia
    stencil = CartesianIndex.([(0,0), (0,1), (1,1), (1,0)])
    ```
    Don't forget to include the zero offset index if you want to include the point itself,
    which is almost always the case.
    Here the stencil creates a 2x2 square extending to the bottom and right of the pixel
    (directions here correspond to the way Julia prints matrices by default).
    When passing a stencil as a vector of `CartesianIndex`, `m = length(stencil)`.

2. As a `D`-dimensional array (where `D` matches the dimensionality of the input data)
    containing `0`s and `1`s, where if `stencil[index] == 1`, the corresponding pixel is
    included, and if `stencil[index] == 0`, it is not included.
    To generate the same estimator as in 1., use

    ```julia
    stencil = [1 1; 1 1]
    ```
    When passing a stencil as a `D`-dimensional array, `m = sum(stencil)`

3. As a `Tuple` containing two `Tuple`s, both of length `D`, for `D`-dimensional data.
    The first tuple specifies the `extent` of the stencil, where `extent[i]`
    dictates the number of pixels to be included along the `i`th axis and `lag[i]`
    the separation of pixels along the same axis.
    This method can only generate (hyper)rectangular stencils. To create the same estimator as
    in the previous examples, use here

    ```julia
    stencil = ((2, 2), (1, 1))
    ```
    When passing a stencil using `extent` and `lag`, `m = prod(extent)!`.

After having defined `est`, one calculates the spatial permutation entropy
by calling [`entropy`](@ref) with `est`, and with the array data.
To apply this to timeseries of spatial data, simply loop over the call, e.g.:

```julia
h = entropy(x, est)
h_vs_t = entropy.(data, est) # broadcasting with `.`
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
struct SpatialSymbolicPermutation{D,P,V} <: SpatialProbEst{D, P}
    stencil::Vector{CartesianIndex{D}}
    viewer::Vector{CartesianIndex{D}}
    arraysize::Dims{D}
    valid::V
end
function SpatialSymbolicPermutation(stencil, x::AbstractArray{T, D};
        periodic::Bool = true) where {T, D}
    stencil, arraysize, valid = preprocess_spatial(stencil, x, periodic)

    SpatialSymbolicPermutation{D, periodic, typeof(valid)}(
        stencil, copy(stencil), arraysize, valid
    )
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

function alphabet_length(est::SpatialSymbolicPermutation)
    m = stencil_length(est.stencil)
    return factorial(m)
end
