using Statistics
export SpatialDispersion
import Base.maximum

"""
    SpatialDispersion <: ProbabilitiesEstimator
    SpatialDispersion(stencil, x::AbstractArray;
        periodic::Bool = true,
        encoding::S = GaussianCDFEncoding(c = 5),
        skip_encoding::Bool = false,
        L = nothing,
    )

A dispersion-based probabilities/entropy estimator for `N`-dimensional spatiotemporal
systems, based on Azami et al. (2019)'s 2D square dispersion entropy estimator,
but here generalized for `N`-dimensional input data `x`.

The argument `periodic` decides whether the stencil should wrap around at the end of the
array. If `periodic = false`, pixels whose stencil exceeds the array bounds are skipped.

## Description

Estimating probabilities/entropies from higher-dimensional data is conceptually simple.

1. Discretize each value (hypervoxel) in `x` relative to all other values `xᵢ ∈ x` using the
    provided `encoding` scheme. If `skip_encoding == true`, `encoding` is
    ignored, and dispersion patterns are computed directly from `x`, under the assumption
    that `L` is the alphabet length for `x` (useful for categorical or integer data).
2. Use `stencil` to extract relevant discretized points around each hypervoxel. The
    `stencil` defines what local area (hyperrectangle), or which points within this area,
    to include around each hypervoxel (i.e. pixel in 2D) (see
    [`SpatioTemporalPermutation`](@ref) for details).
3. Construct a symbol string from these points.
4. Take the sum-normalized histogram of the symbol strings as a probability distribution.
5. Optionally, compute [`entropy`](@ref) or [`entropy_normalized`](@ref) from this
    probability distribution.

## Application on time series

To apply this to timeseries of spatial data, simply loop over the call (broadcast), e.g.:

```julia
imgs = [rand(50, 50) for i = 1:100]; # one image per second over 100 seconds
stencil = ((2, 2), (1, 1)) # a 2x2 stencil (i.e. dispersion patterns of length 4)
est = SpatialDispersion(stencil, first(imgs))
h_vs_t = entropy_normalized.(imgs, Ref(est))
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
    encoding::S
    skip_encoding::Bool
    L::Union{Nothing, Int}
end

function SpatialDispersion(stencil, x::AbstractArray{T, D};
        periodic::Bool = true,
        encoding::S = GaussianCDFEncoding(c = 5),
        skip_encoding::Bool = false,
        L::Union{Nothing, Int} = nothing) where {S, T, D}
    stencil, arraysize, valid = preprocess_spatial(stencil, x, periodic)

    SpatialDispersion{D, periodic, typeof(valid), S}(
        stencil, copy(stencil), arraysize, valid, encoding,
        skip_encoding, L,
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

function symbol_distribution(x::AbstractArray{T, N}, est::SpatialDispersion) where {T, N}
    if est.skip_encoding
        symbolized_x = copy(x)
    else
        # Symbolize each pixel individually relative to the other pixels.
        # This will be an integer array with the same dimensions as `x`.
        symbolized_x = outcomes(x, est.encoding)
    end

    # It is easiest just to store the symbols as strings, i.e. [1, 5, 4, 3] => "1543".
    # TODO: potential optimization:
    # - represent each dispersion pattern as SVector{Int}? (requires conversion)
    # - represent each dispersion pattern as Int (requires D multiplications) per pattern?
    # - maybe strings are fast enough, so no optimization needed?
    symbols = Vector{String}(undef, 0)
    for pixel in est.valid
        pixels_inds = pixels_in_stencil(pixel, est)
        push!(symbols, join(view(symbolized_x, pixels_inds)))
    end
    return symbols
end

function probabilities(x::AbstractArray{T, N}, est::SpatialDispersion) where {T, N}
    symbols = symbol_distribution(x, est)
    return Probabilities(fasthist!(symbols))
end

function probabilities_and_outcomes(x::Array_or_Dataset, est::SpatialDispersion)
    symbols = symbol_distribution(x, est)

    # We don't care about the fact that `fasthist!` sorts in-place here, because we
    # only need the unique values of `symbols` for the outcomes.
    probs = Probabilities(fasthist!(symbols))
    outcomes = unique!(symbols)
    return probs, outcomes
end

function alphabet_length(est::SpatialDispersion)
    m = stencil_length(est.stencil)
    if est.skip_encoding
        return est.L^m
    else
        return est.encoding.c^m
    end
end
