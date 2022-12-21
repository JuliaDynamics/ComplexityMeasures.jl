using Statistics
export SpatialDispersion
import Base.maximum

"""
    SpatialDispersion <: ProbabilitiesEstimator
    SpatialDispersion(stencil, x::AbstractArray;
        periodic = true,
        c = 5,
        skip_encoding = false,
        L = nothing,
    )

A dispersion-based probabilities/entropy estimator for `N`-dimensional spatiotemporal
systems, based on Azami et al. (2019)'s 2D square dispersion entropy estimator,
but here generalized for `N`-dimensional input data `x`.

## Arguments

- `stencil`. Defines what local area (hyperrectangle), or which points within this area,
    to include around each hypervoxel (i.e. pixel in 2D). The examples below demonstrate
    different ways of specifying stencils. For details, see
    [`SpatialSymbolicPermutation`](@ref).
-  `x::AbstractArray`. The input data. Must be provided because we need to know its size
    for optimization and bound checking.

## Keyword arguments

- `periodic::Bool`. If `periodic == true`, then the stencil should wrap around at the
    end of the array. If `periodic = false`, then pixels whose stencil exceeds the array
    bounds are skipped.
- `encoding::Encoding`. Determines how input data is mapped to discrete categories. Must be
    a valid [`Encoding`](@ref).
- `skip_encoding`. If `skip_encoding == true`, `encoding` is ignored, and dispersion
    patterns are computed directly from `x`, under the assumption that `L` is the alphabet
    length for `x` (useful for categorical or integer data). Thus, if
    `skip_encoding == true`, then `L` must also be specified. This is useful for
    categorical or integer-valued data.
- `L`. If `L == nothing` (default), then the number of total outcomes is inferred from
    `stencil` and `encoding`. If `L` is set to an integer, then the data is considered
    pre-encoded and the number of total outcomes is set to `L`.

## Description

Estimating probabilities/entropies from higher-dimensional data is conceptually simple.

1. Discretize each value (hypervoxel) in `x` relative to all other values `xᵢ ∈ x` using the
    provided `encoding` scheme.
2. Use `stencil` to extract relevant (discretized) points around each hypervoxel.
3. Construct a symbol string from these points.
4. Take the sum-normalized histogram of the symbol strings as a probability distribution.
5. Optionally, compute [`entropy`](@ref) or [`entropy_normalized`](@ref) from this
    probability distribution.

## Usage

Here's how to compute spatial dispersion entropy using the three different ways of
specifying stencils.

```julia
x = rand(50, 50) # first "time slice" of a spatial system evolution

# Cartesian stencil
stencil_cartesian = CartesianIndex.([(0,0), (1,0), (1,1), (0,1)])
est = SpatialDispersion(stencil_cartesian, x)
entropy_normalized(est, x)

# Extent/lag stencil
extent = (2, 2); lag = (1, 1); stencil_ext_lag = (extent, lag)
est = SpatialDispersion(stencil_ext_lag, x)
entropy_normalized(est, x)

# Matrix stencil
stencil_matrix = [1 1; 1 1]
est = SpatialDispersion(stencil_matrix, x)
entropy_normalized(est, x)
```

To apply this to timeseries of spatial data, simply loop over the call (broadcast), e.g.:

```julia
imgs = [rand(50, 50) for i = 1:100]; # one image per second over 100 seconds
stencil = ((2, 2), (1, 1)) # a 2x2 stencil (i.e. dispersion patterns of length 4)
est = SpatialDispersion(stencil, first(imgs))
h_vs_t = entropy_normalized.(Ref(est), imgs)
```

See also: [`SpatialSymbolicPermutation`](@ref), [`GaussianCDFEncoding`](@ref),
[`symbolize`](@ref).

[^Azami2019]: Azami, H., da Silva, L. E. V., Omoto, A. C. M., & Humeau-Heurtier, A. (2019).
    Two-dimensional dispersion entropy: An information-theoretic method for irregularity
    analysis of images. Signal Processing: Image Communication, 75, 178-187.
"""
struct SpatialDispersion{D,P,V,S<:Encoding} <: SpatialProbEst{D, P}
    stencil::Vector{CartesianIndex{D}}
    viewer::Vector{CartesianIndex{D}}
    arraysize::Dims{D}
    valid::V
    c::Int
    encoding::Type{S}
    skip_encoding::Bool
    L::Union{Nothing, Int}
    m::Int
end

function SpatialDispersion(stencil, x::AbstractArray{T, D};
        periodic::Bool = true,
        c::Int = 5,
        encoding::Type{S} = GaussianCDFEncoding,
        skip_encoding::Bool = false,
        L::Union{Nothing, Int} = nothing) where {S <: Encoding, T, D}
    stencil, arraysize, valid = preprocess_spatial(stencil, x, periodic)
    if skip_encoding
        !isnothing(L) || throw(
            ArgumentError("When `skip_encoding == true`, `L` must be an integer.")
        )
    end
    m = stencil_length(stencil)

    SpatialDispersion{D, periodic, typeof(valid), S}(
        stencil, copy(stencil), arraysize, valid, c, encoding,
        skip_encoding, L, m,
    )
end

# Pretty printing
function Base.show(io::IO, est::SpatialDispersion{D,P,V,S}) where {D,P,V,S}
    println(io, "Spatial dispersion estimator for $D-dimensional data.")
    print(io, "Stencil: ")
    show(io, MIME"text/plain"(), est.stencil)
    print(io, "\nEncoding: $(est.encoding)")
    print(io, """\nBoundaries: $(P ? "Periodic" : "Non-periodic")""")
end

function symbol_distribution(est::SpatialDispersion, x::AbstractArray{T, N}) where {T, N}
    # This looks funny, but E is just a type representing some encoding that accept
    # the keyword argument `c`, which dictates how many discrete categories to use,
    # and the mandatory keyword arguments `μ` and `σ`, which are the empirical mean
    # and standard deviation. The code on the following two lines assumes
    # that the encoding is always a GaussianCDFEncoding, but it doesn't need to be.
    E = est.encoding
    encoding = E(; c = est.c, μ = mean(x), σ = std(x))
    if est.skip_encoding
        encoded_x = copy(x)
    else
        # Symbolize each pixel individually relative to the other pixels.
        # This will be an integer array with the same dimensions as `x`.
        encoded_x = encode.(Ref(encoding), x)
    end

    symbols = Vector{SVector{est.m, Int}}(undef, 0)
    for pixel in est.valid
        pixels_inds = pixels_in_stencil(est, pixel)
        s = SVector{est.m, Int}(view(encoded_x, pixels_inds))
        push!(symbols, s)
    end
    return symbols
end

function probabilities(est::SpatialDispersion, x::AbstractArray{T, N}) where {T, N}
    symbols = symbol_distribution(est, x)
    return Probabilities(fasthist!(symbols))
end

function probabilities_and_outcomes(est::SpatialDispersion, x::AbstractArray{T, N}) where {T, N}
    symbols = symbol_distribution(est, x)

    # We don't care about the fact that `fasthist!` sorts in-place here, because we
    # only need the unique values of `symbols` for the outcomes.
    probs = Probabilities(fasthist!(symbols))
    outcomes = unique!(symbols)
    return probs, outcomes
end

function total_outcomes(est::SpatialDispersion)::Int
    m = est.m
    if est.skip_encoding
        return est.L^m
    else
        c = est.c
        return c^m
    end
end

# TODO: how to represent the outcomes? We have to think about this...
function outcome_space(est::SpatialDispersion)

end
