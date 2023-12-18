using StaticArrays: @MVector

export SpatialOrdinalPatterns
###########################################################################################
# type creation
###########################################################################################

"""
    SpatialOrdinalPatterns <: OutcomeSpaceModel
    SpatialOrdinalPatterns(stencil, x; periodic = true)

A symbolic, permutation-based [`OutcomeSpace`](@ref) for spatiotemporal systems that
generalises [`OrdinalPatterns`](@ref) to high-dimensional arrays.
The order `m` of the permutation pattern is extracted from the `stencil`, see below.

`SpatialOrdinalPatterns` is based on the 2D and 3D *spatiotemporal permutation entropy*
estimators by [Ribeiro2012](@citet) and [Schlemmer2018](@citet), respectively, but is here
implemented as a pure probabilities probabilities estimator that is generalized for
`D`-dimensional input array `x`, with arbitrary regions (stencils) to get patterns form
and (possibly) periodic boundary conditions.

See below for ways to specify the `stencil`. If `periodic = true`, then the stencil wraps
around at the ends of the array. If `false`, then collected regions with indices
which exceed the array bounds are skipped.

In combination with [`information`](@ref) and [`information_normalized`](@ref), this
probabilities estimator can be used to compute generalized spatiotemporal
permutation [`InformationMeasure`](@ref) of any type.

## Outcome space

The outcome space `Ω` for `SpatialOrdinalPatterns` is the set of length-`m` ordinal
patterns (i.e. permutations) that can be formed by the integers `1, 2, …, m`,
ordered lexicographically. There are `factorial(m)` such patterns.
Here `m` refers to the number of points included in `stencil`.

## Stencils

The `stencil` defines what local area to use to group hypervoxels. Each grouping
of hypervoxels is mapped to an order-`m` permutation pattern,
which is then mapped to an integer as in [`OrdinalPatterns`](@ref).
The `stencil` is moved around the input array, in a sense "scanning" the input array,
to collect all possible groupings allowed by the boundary condition (periodic or not).

Stencils are passed in one of the following three ways:

1. As vectors of `CartesianIndex` which encode the offset of indices to include in the
    stencil, with respect to the current array index when scanning over the array.
    For example `stencil = CartesianIndex.([(0,0), (0,1), (1,1), (1,0)])`.
    Don't forget to include the zero offset index if you want to include the hypervoxel
    itself, which is almost always the case.
    Here the stencil creates a 2x2 square extending to the bottom and right of the pixel
    (directions here correspond to the way Julia prints matrices by default).
    When passing a stencil as a vector of `CartesianIndex`, `m = length(stencil)`.
2. As a `D`-dimensional array (where `D` matches the dimensionality of the input data)
    containing `0`s and `1`s, where if `stencil[index] == 1`, the corresponding pixel is
    included, and if `stencil[index] == 0`, it is not included.
    To generate the same estimator as in 1., use `stencil = [1 1; 1 1]`.
    When passing a stencil as a `D`-dimensional array, `m = sum(stencil)`
3. As a `Tuple` containing two `Tuple`s, both of length `D`, for `D`-dimensional data.
    The first tuple specifies the `extent` of the stencil, where `extent[i]`
    dictates the number of hypervoxels to be included along the `i`th axis and `lag[i]`
    the separation of hypervoxels along the same axis.
    This method can only generate (hyper)rectangular stencils. To create the same estimator as
    in the previous examples, use here `stencil = ((2, 2), (1, 1))`.
    When passing a stencil using `extent` and `lag`, `m = prod(extent)`.
"""
struct SpatialOrdinalPatterns{D,P,V,M,F} <: SpatialOutcomeSpace{D, P}
    stencil::Vector{CartesianIndex{D}}
    viewer::Vector{CartesianIndex{D}}
    arraysize::Dims{D}
    valid::V
    encoding::OrdinalPatternEncoding{M,F}
end

function SpatialOrdinalPatterns(stencil, x::AbstractArray{T, D};
        periodic::Bool = true, lt::F = isless_rand) where {T, D, F}
    stencil, arraysize, valid = preprocess_spatial(stencil, x, periodic)
    m = stencil_length(stencil)
    encoding = OrdinalPatternEncoding{m}(lt)
    return SpatialOrdinalPatterns{D,periodic,typeof(valid),m,F}(
        stencil, copy(stencil), arraysize, valid, encoding
    )
end

function encoded_space_cardinality(est::SpatialOrdinalPatterns, x::AbstractArray{T, N}) where {T, N}
    s = zeros(Int, length(est.valid))
    encodings_from_permutations!(s, est, x)
    return length(s)
end

function counts_and_outcomes(est::SpatialOrdinalPatterns, x)
    s = zeros(Int, length(est.valid))
    return counts_and_outcomes!(s, est, x)
end

function counts_and_outcomes!(s, est::SpatialOrdinalPatterns, x)
    encodings_from_permutations!(s, est, x)
    observed_outcomes = decode.(Ref(est.encoding), s)
    outs = sort!(unique(observed_outcomes))
    z = copy(s)
    cts = fasthist!(z)
    c = Counts(cts, (outs, ))
    return c, outcomes(c)
end

# Don't use generic dispatch, because we need to use `counts_and_outcomes`!.
function probabilities!(est::SpatialOrdinalPatterns, x, s)
    s = zeros(Int, length(est.valid))
    return Probabilities(first(counts_and_outcomes!(s, est, x)))
end

# Pretty printing
function Base.show(io::IO, est::SpatialOrdinalPatterns{D,P,V,M}) where {D,P,V,M}
    print(io, "Spatial symbolic permutation probabilities estimator"*
              "of order $(M) and for $D-dimensional data. Periodic: $(P). Stencil:")
    print(io, "\n")
    show(io, MIME"text/plain"(), est.stencil)
end

outcome_space(est::SpatialOrdinalPatterns) = outcome_space(est.encoding)
total_outcomes(est::SpatialOrdinalPatterns) = total_outcomes(est.encoding)

function encodings_from_permutations!(πs, est::SpatialOrdinalPatterns, x::AbstractArray)
    check_preallocated_length!(πs, est, x)
    for (i, pixel) in enumerate(est.valid)
        pixels = pixels_in_stencil(est, pixel)
        selection = view(x, pixels)
        πs[i] = encode(est.encoding, selection)
    end
    return πs
end

function check_preallocated_length!(
        πs, est::SpatialOrdinalPatterns{D, periodic}, x::AbstractArray{T, N}
    ) where {D, periodic, T, N}
    if periodic
        # If periodic boundary conditions, then each pixel has a well-defined neighborhood,
        # and there are as many encoded symbols as there are pixels.
        length(πs) == length(x) ||
            throw(
                ArgumentError(
                    """Need length(πs) == length(x), got `length(πs)=$(length(πs))`
                    and `length(x)==$(length(x))`."""
                )
            )
    else
        # If not periodic, then we must count the number of encoded symbols from the
        # valid coordinates of the estimator.
        length(πs) == length(est.valid) ||
        throw(
            ArgumentError(
                """Need length(πs) == length(est.valid), got `length(πs)=$(length(πs))`
                and `length(est.valid)==$(length(est.valid))`."""
            )
        )
    end
end
