export SpatialSymbolicPermutation

"""
    SpatialSymbolicPermutation(stencil, x; periodic = true)

A symbolic, permutation-based probabilities/entropy estimator for spatiotemporal systems.
The input data `x` are high-dimensional arrays, for example 2D arrays [^Ribeiro2012]
(e.g. images) or 3D arrays [^Schlemmer2018] (3D images). This approach is also known as
_spatiotemporal permutation entropy_.

## Arguments

- `stencil`. Defines what local area (which points) around each pixel to
    consider, and compute ordinal patterns from. The number of included points in a
    stencil (`m`) determines the length of the vectors to be symbolized, i.e. there are
    `m!` possible ordinal patterns around each pixel. Stencils are passed in one of the
    following three ways:

    1. As vectors of `CartesianIndex` which encode the pixels to include in the
        stencil, with respect to the current pixel, or integer arrays of the same
        dimensionality as the data. For example
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
        This method can only generate (hyper)rectangular stencils. To create the same
        estimator as in the previous examples, use here
        ```julia
        stencil = ((2, 2), (1, 1))
        ```
        When passing a stencil using `extent` and `lag`, `m = prod(extent)!`.

- `x::AbstractArray`. The input data. Must be provided because we need to know its size
    for optimization and bound checking.

## Keyword arguments

- `periodic::Bool`. If `periodic == true`, then the stencil should wrap around at the
    end of the array. If `periodic = false`, then pixels whose stencil exceeds the array
    bounds are skipped.

## Usage

```julia
data = [rand(50, 50) for _ in 1:50]
x = data[1] # first "time slice" of a spatial system evolution

# Cartesian stencil
stencil_cartesian = CartesianIndex.([(0,0), (1,0), (1,1), (0,1)])
est = SpatialSymbolicPermutation(stencil_cartesian, x)

# Extent/lag stencil
extent = (2, 2); lag = (1, 1)
stencil_ext_lag = (extent, lag)
est = SpatialSymbolicPermutation(stencil_ext_lag, x)

# Matrix stencil
stencil_matrix = [1 1; 1 1]
est = SpatialSymbolicPermutation(stencil_matrix, x)
```

###
To apply this to timeseries of spatial data, simply loop over the call, e.g.:
```julia
h = entropy(x, est)
h_vs_t = [entropy(d, est) for d in data]
```

See also: [`SpatialDispersion`](@ref).

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
    m::Int
end

function SpatialSymbolicPermutation(stencil, x::AbstractArray{T, D};
        periodic::Bool = true) where {T, D}
    stencil, arraysize, valid = preprocess_spatial(stencil, x, periodic)
    m = stencil_length(stencil)

    SpatialSymbolicPermutation{D, periodic, typeof(valid)}(
        stencil, copy(stencil), arraysize, valid, m
    )
end

function probabilities(x, est::SpatialSymbolicPermutation)
    # TODO: This can be literally a call to `symbolize` and then
    # calling probabilities on it. Should do once the `symbolize` refactoring is done.
    s = zeros(Int, length(est.valid))
    probabilities!(s, x, est)
end

function probabilities!(s, x, est::SpatialSymbolicPermutation)
    m = length(est.stencil)
    for (i, pixel) in enumerate(est.valid)
        pixels = pixels_in_stencil(pixel, est)
        s[i] = encode_motif(view(x, pixels), m)
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
    return factorial(est.m)
end
