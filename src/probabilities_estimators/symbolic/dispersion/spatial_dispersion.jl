using Statistics
export SpatialDispersion
import Base.maximum

"""
    SpatialDispersion <: ProbabilitiesEstimator
    SpatialDispersion(stencil, x::AbstractArray;
        periodic::Bool = true,
        symbolization::S = GaussianSymbolization(c = 5),
        skip_symbolization::Bool = false,
        L = nothing,
    )

A dispersion-based probabilities/entropy estimator for `N`-dimensional spatiotemporal
systems, based on Azami et al. (2019)'s 2D square dispersion entropy estimator,
but here generalized for `N`-dimensional input data `x`, as well as arbitrary `stencil`s
and `symbolization` schemes.

The `symbolization` scheme dictates how raw data are symbolized, while `stencil` defines
what local area (hyperrectangle), or which points within this area, to include around each
hypervoxel (i.e. pixel in 2D) (see [`SpatioTemporalPermutation`](@ref) for details).

If `skip_symbolization == true`, `symbolization` is ignored, and dispersion patterns are
computed directly from `x`, under the assumption that `L` is the alphabet length for `x`
(useful for categorical or integer data).

The argument `periodic` decides whether the stencil should wrap around at the end of the
array. If `periodic = false`, pixels whose stencil exceeds the array bounds are skipped.

## Description

Estimating probabilities/entropies from higher-dimensional data is conceptually simple.
The steps are

1. Discretize each value (hypervoxel) in `x` relative to all other values `xᵢ ∈ x` using the
    provided `symbolization` scheme.
2. Use `stencil` to extract relevant discretized points around each hypervoxel.
3. Construct a symbol string from these points.
4. Take the sum-normalized histogram of the symbol strings as a probability distribution.
5. Optionally, compute entropy from this probability distribution.

## Application on time series

To apply this to timeseries of spatial data, simply loop over the call (broadcast), e.g.:

```julia
imgs = [rand(50, 50) for i = 1:100]; # one image per second over 100 seconds
stencil = ((2, 2), (1, 1)) # a 2x2 stencil (i.e. dispersion patterns of length 4)
est = SpatialDispersion(stencil, first(imgs))
h_vs_t = entropy_normalized.(imgs, Ref(est))
```

See also: [`SpatioTemporalPermutation`](@ref), [`GaussianSymbolization`](@ref),
[`symbolize`](@ref).

[^Azami2019]: Azami, H., da Silva, L. E. V., Omoto, A. C. M., & Humeau-Heurtier, A. (2019).
    Two-dimensional dispersion entropy: An information-theoretic method for irregularity
    analysis of images. Signal Processing: Image Communication, 75, 178-187.
[^Furlong2021]: Furlong, R., Hilal, M., O’brien, V., & Humeau-Heurtier, A. (2021).
    Parameter Analysis of Multiscale Two-Dimensional Fuzzy and Dispersion Entropy
    Measures Using Machine Learning Classification. Entropy, 23(10), 1303.
"""
struct SpatialDispersion{D,P,V,S<:SymbolizationScheme} <: ProbabilitiesEstimator
    stencil::Vector{CartesianIndex{D}}
    viewer::Vector{CartesianIndex{D}}
    arraysize::Dims{D}
    valid::V
    symbolization::S
    periodic::Bool
    skip_symbolization::Bool
    L::Union{Nothing, Int}
end

function SpatialDispersion(stencil, x::AbstractArray;
        periodic::Bool = true,
        symbolization::S = GaussianSymbolization(c = 5),
        skip_symbolization::Bool = false,
        L::Union{Nothing, Int} = nothing) where S
    stencil, arraysize, valid, D = preprocess_spatial(stencil, x, periodic)

    SpatialDispersion{D, periodic, typeof(valid), S}(
        stencil, copy(stencil), arraysize, valid, symbolization, periodic,
        skip_symbolization, L,
    )
end

# Pretty printing
function Base.show(io::IO, est::SpatialDispersion{D}) where {D}
    println(io, "Spatial dispersion estimator for $D-dimensional data.")
    print(io, "Stencil: ")
    show(io, MIME"text/plain"(), est.stencil)
    print(io, "\nSymbolization: $(est.symbolization)")
    print(io, """\nBoundaries: $(est.periodic ? "Periodic" : "Non-periodic")""")
end

pixels_in_stencil(pixel, est::SpatialDispersion{D,false}) where {D} =
    get_pixels_nonperiodic(pixel, est)

pixels_in_stencil(pixel, est::SpatialDispersion{D,true}) where {D} =
    get_pixels_periodic(pixel, est, D)

function symbol_distribution(x::AbstractArray{T, N}, est::SpatialDispersion) where {T, N}
    if est.skip_symbolization
        symbolized_x = copy(x)
    else
        # Symbolize each pixel individually relative to the other pixels.
        # This will be an integer array with the same dimensions as `x`.
        symbolized_x = symbolize(x, est.symbolization)
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

function probabilities_and_events(x::Array_or_Dataset, est::SpatialDispersion)
    symbols = symbol_distribution(x, est)

    # We don't care about the fact that `fasthist!` sorts in-place here, because we
    # only need the unique values of `symbols` for the events.
    probs = Probabilities(fasthist!(symbols))
    events = unique!(symbols)
    return probs, events
end

function alphabet_length(est::SpatialDispersion)
    m = stencil_length(est.stencil)
    if est.skip_symbolization
        return est.L^m
    else
        return est.symbolization.c^m
    end
end
