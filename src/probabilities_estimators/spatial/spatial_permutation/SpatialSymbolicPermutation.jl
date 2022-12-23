using StaticArrays: @MVector

export SpatialSymbolicPermutation
###########################################################################################
# type creation
###########################################################################################

"""
    SpatialSymbolicPermutation <: ProbabilitiesEstimator
    SpatialSymbolicPermutation(stencil, x; periodic = true)

A symbolic, permutation-based probabilities estimator for spatiotemporal systems that
generalises [`SymbolicPermutation`](@ref) to high-dimensional arrays.
The order `m` of the permutation pattern is extracted from the `stencil`, see below.

`SpatialSymbolicPermutation` is based on the 2D and 3D *spatiotemporal permutation entropy*
estimators by by Ribeiro et al. (2012)[^Ribeiro2012] and Schlemmer et al.
(2018)[^Schlemmer2018]), respectively, but is here implemented as a pure probabilities
probabilities estimator that is generalized for `D`-dimensional input array `x`,
with arbitrary regions (stencils) to get patterns form
and (possibly) periodic boundary conditions.

See below for ways to specify the `stencil`. If `periodic = true`, then the stencil wraps
around at the ends of the array. If `false`, then collected regions with indices
which exceed the array bounds are skipped.

In combination with [`entropy`](@ref) and [`entropy_normalized`](@ref), this
probabilities estimator can be used to compute generalized spatiotemporal
permutation [`Entropy`](@ref) of any type.

## Outcome space

The outcome space `Ω` for `SpatialSymbolicPermutation` is the set of length-`m` ordinal
patterns (i.e. permutations) that can be formed by the integers `1, 2, …, m`,
ordered lexicographically. There are `factorial(m)` such patterns.
Here `m` refers to the number of points included in `stencil`.

## Stencils

The `stencil` defines what local area to use to group hypervoxels. Each grouping
of hypervoxels is mapped to an order-`m` permutation pattern,
which is then mapped to an integer as in [`SymbolicPermutation`](@ref).
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
    lt::Function
    m::Int
end

function SpatialSymbolicPermutation(stencil, x::AbstractArray{T, D};
        periodic::Bool = true, lt = isless_rand) where {T, D}
    stencil, arraysize, valid = preprocess_spatial(stencil, x, periodic)
    m = stencil_length(stencil)

    SpatialSymbolicPermutation{D, periodic, typeof(valid)}(
        stencil, copy(stencil), arraysize, valid, lt, m
    )
end

function probabilities(est::SpatialSymbolicPermutation, x)
    # TODO: This can be literally a call to `symbolize` and then
    # calling probabilities on it. Should do once the `symbolize` refactoring is done.
    s = zeros(Int, length(est.valid))
    probabilities!(s, est, x)
end

function probabilities!(s, est::SpatialSymbolicPermutation, x)
    encodings_from_permutations!(s, est, x)
    return probabilities(s)
end

function probabilities_and_outcomes(est::SpatialSymbolicPermutation, x)
    # TODO: This can be literally a call to `symbolize` and then
    # calling probabilities on it. Should do once the `symbolize` refactoring is done.
    s = zeros(Int, length(est.valid))
    probabilities_and_outcomes!(s, est, x)
end

function probabilities_and_outcomes!(s, est::SpatialSymbolicPermutation, x)
    m, lt = est.m, est.lt
    encoding = OrdinalPatternEncoding(; m, lt)

    encodings_from_permutations!(s, est, x)
    observed_outcomes = decode.(Ref(encoding), s)
    return probabilities(s), observed_outcomes
end

# Pretty printing
function Base.show(io::IO, est::SpatialSymbolicPermutation{D}) where {D}
    print(io, "Spatial permutation estimator for $D-dimensional data. Stencil:")
    print(io, "\n")
    show(io, MIME"text/plain"(), est.stencil)
end

function outcome_space(est::SpatialSymbolicPermutation)
    encoding = OrdinalPatternEncoding(; est.m, est.lt)
    decode.(Ref(encoding), 1:factorial(est.m))
end

function total_outcomes(est::SpatialSymbolicPermutation)
    return factorial(est.m)
end

function check_preallocated_length!(πs, est::SpatialSymbolicPermutation{D, periodic}, x::AbstractArray{T, N}) where {D, periodic, T, N}
    if periodic
        # If periodic boundary conditions, then each pixel has a well-defined neighborhood,
        # and there are as many encoded symbols as there are pixels.
        length(πs) == length(x) ||
            throw(
                ArgumentError(
                    """Need length(πs) == length(x), got `length(πs)=$(length(πs))`\
                    and `length(x)==$(length(x))`."""
                )
            )
    else
        # If not periodic, then we must count the number of encoded symbols from the
        # valid coordinates of the estimator.
        length(πs) == length(est.valid) ||
        throw(
            ArgumentError(
                """Need length(πs) == length(est.valid), got `length(πs)=$(length(πs))`\
                and `length(est.valid)==$(length(est.valid))`."""
            )
        )
    end
end

function encodings_from_permutations!(πs, est::SpatialSymbolicPermutation{D, periodic},
        x::AbstractArray{T, N}) where {T, N, D, periodic}
    m, lt = est.m, est.lt
    check_preallocated_length!(πs, est, x)
    encoding = OrdinalPatternEncoding(; m, lt)

    perm = @MVector zeros(Int, m)
    for (i, pixel) in enumerate(est.valid)
        pixels = pixels_in_stencil(est, pixel)
        sortperm!(perm, view(x, pixels)) # Find permutation for currently selected pixels.
        πs[i] = encode(encoding, perm) # Encode based on the permutation.
    end
    return πs
end
