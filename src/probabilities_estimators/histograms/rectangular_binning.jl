export RectangularBinning, FixedRectangularBinning
export RectangularBinEncoding

abstract type AbstractBinning end
abstract type HistogramEncoding <: Encoding end

##################################################################
# Structs and docstrings
##################################################################
# Notice that the binning types are intermediate structs that are NOT retained
# in the source code. Their only purpose is instructions of how to create a
# `RectangularBinEncoder`. All actual source code functionality of `ValueHistogram`
# is implemented based on `RectangularBinEncoder`.

"""
    RectangularBinning(ϵ) <: AbstractBinning

Rectangular box partition of state space using the scheme `ϵ`,
deducing the coordinates of the grid axis minima from the input data.

Binning instructions are deduced from the type of `ϵ` as follows:

1. `ϵ::Int` divides each coordinate axis into `ϵ` equal-length intervals
    that cover all data.
2. `ϵ::Float64` divides each coordinate axis into intervals of fixed size `ϵ`, starting
    from the axis minima until the data is completely covered by boxes.
3. `ϵ::Vector{Int}` divides the i-th coordinate axis into `ϵ[i]` equal-length
    intervals that cover all data.
4. `ϵ::Vector{Float64}` divides the i-th coordinate axis into intervals of fixed size
    `ϵ[i]`, starting from the axis minima until the data is completely covered by boxes.
"""
struct RectangularBinning{E} <: AbstractBinning
    ϵ::E
end

const ValidFixedBinInputs = Union{Number, NTuple}

"""
    FixedRectangularBinning <: AbstractBinning
    FixedRectangularBinning(ϵmin::E, ϵmax::E, N::Int) where E

Rectangular box partition of state space where the extent of the grid is explicitly
specified by `ϵmin` and `emax`, and along each dimension, the grid is subdivided into `N`
subintervals.

Binning instructions are deduced from the types of `ϵmin`/`emax` as follows:

1. `E<:Float64` sets the grid range along all dimensions to to `[ϵmin, ϵmax]`.
2. `E::NTuple{D, Float64}` sets ranges along each dimension individually, i.e.
    `[ϵmin[i], ϵmax[i]]` is the range along the `i`-th dimension.

If the grid spans the range `[r1, r2]` along a particular dimension, then this range
is subdivided into `N` subintervals of equal length `nextfloat((r2 - r1)/N)`.
Thus, for `D`-dimensional data, there are `N^D` boxes.
"""
struct FixedRectangularBinning{E} <: AbstractBinning
    ϵmin::E
    ϵmax::E
    N::Int

    function FixedRectangularBinning(ϵmin::E1, ϵmax::E2, N::Int) where {E1 <: ValidFixedBinInputs, E2 <: ValidFixedBinInputs}
        f_ϵmin = float.(ϵmin)
        f_ϵmax = float.(ϵmax)
        return new{typeof(f_ϵmin)}(f_ϵmin, f_ϵmax, N::Int)
    end
end

const Floating_or_Fixed_RectBinning = Union{RectangularBinning, FixedRectangularBinning}

"""
    RectangularBinEncoding <: Encoding
    RectangularBinEncoding(x, binning::AbstractBinning)
    RectangularBinEncoding(binning::FixedRectangularBinning{<:NTuple{D}})

Struct used in [`outcomes`](@ref) to map points of `x` into their respective bins.
It finds the minima along each dimension, and computes appropriate
edge lengths for each dimension of `x` given a rectangular binning.

The second signature does not need `x` because (1) the binning is fixed, and the
size of `x` doesn't matter, and (2) because the binning contains the dimensionality
information as `ϵmin/max` is already an `NTuple`.

See also: [`RectangularBinning`](@ref), [`FixedRectangularBinning`](@ref).
"""
struct RectangularBinEncoding{B, D, T, C, L} <: HistogramEncoding
    binning::B # either RectangularBinning or FixedRectangularBinning
    mini::SVector{D,T} # fields are either static vectors or numbers
    edgelengths::SVector{D,T}
    histsize::SVector{D,Int}
    ci::C # cartesian indices
    li::L # linear indices
end

function Base.show(io::IO, x::RectangularBinEncoding)
    return print(io, "RectangularBinEncoding\n" *
        "  binning: $(x.binning) \n" *
        "  box corners: $(x.mini)\n" *
        "  edgelengths: $(x.edgelengths)"
    )
end

function encode(point, e::RectangularBinEncoding)
    (; mini, edgelengths) = e
    # Map a data point to its bin edge (plus one because indexing starts from 1)
    bin = floor.(Int, (point .- mini) ./ edgelengths) .+ 1
    return e.li[CartesianIndex(Tuple(bin))]
end

function decode(bin::Int, e::RectangularBinEncoding{B, V}) where {B, V}
    cartesian = e.ci[bin]
    (; mini, edgelengths) = e
    # Remove one because we want lowest value corner, and we get indices starting from 1
    return (V(Tuple(cartesian)) .- 1) .* edgelengths .+ mini
end

##################################################################
# Initialization of encodings
##################################################################
# Data-controlled grid
function RectangularBinEncoding(x, b::RectangularBinning; n_eps = 2)
    # This function always returns static vectors and is type stable
    D = dimension(x)
    T = eltype(x)
    ϵ = b.ϵ
    mini, maxi = minmaxima(x)
    v = ones(SVector{D,T})
    if ϵ isa Float64 || ϵ isa AbstractVector{<:AbstractFloat}
        edgelengths = SVector{D,T}(ϵ .* v)
        histsize = round.(Int, (maxi .- mini) ./ edgelengths)
    elseif ϵ isa Int || ϵ isa Vector{Int}
        edgeslengths_nonadjusted = @. (maxi - mini)/ϵ
        # Just taking nextfloat once here isn't enough for bins to cover data when using
        # `encode_as_bin` later, because subtraction and division leads to loss
        # of precision. We need a slightly bigger number, so apply nextfloat twice.
        edgelengths = SVector{D,T}(nextfloat.(edgeslengths_nonadjusted, n_eps))
        if ϵ isa Vector{Int}
            histsize = SVector{D, Int}(ϵ)
        else
            histsize = SVector{D, Int}(fill(ϵ, D))
        end
    else
        error("Invalid ϵ for binning of a dataset")
    end
    # Cartesian indices of the underlying histogram
    ci = CartesianIndices(Tuple(histsize))
    li = LinearIndices(ci)
    RectangularBinEncoding(b, mini, edgelengths, histsize, ci, li)
end

# fixed grid
function RectangularBinEncoding(x, b::FixedRectangularBinning{E}; n_eps = 2) where {E}
    D = dimension(x)
    T = eltype(x)
    D ≠ length(E) && error("Dimension of data and fixed rectangular binning don't match!")
    # This function always returns static vectors and is type stable
    ϵmin, ϵmax = b.ϵmin, b.ϵmax
    if E <: Real
        mini = SVector{D, T}(repeat([ϵmin], D))
        maxi = SVector{D, T}(repeat([ϵmax], D))
    elseif E <: NTuple{D}
        mini = SVector{D, T}(ϵmin)
        maxi = SVector{D, T}(ϵmax)
    else
        error("Invalid ϵmin or ϵmax for binning of a dataset")
    end
    edgelengths_nonadjusted = @. (maxi .- mini) / b.N
    edgelengths = nextfloat.(edgelengths_nonadjusted, n_eps)
    histsize = SVector{D,Int}(fill(b.N, D))
    ci = CartesianIndices(Tuple(histsize))
    li = LinearIndices(ci)
    RectangularBinEncoding(b, mini, edgelengths, histsize, ci, li)
end

# This version exists if the given `ϵ`s are already tuples.
# Then, the dataset doesn't need to be provided.
function RectangularBinEncoding(b::FixedRectangularBinning{<:NTuple{D,T}}; n_eps = 2) where {D,T}
    return RectangularBinEncoding(Dataset{D,T}(), b; n_eps)
end

##################################################################
# Outcomes / total outcomes
##################################################################
total_outcomes(e::RectangularBinEncoding) = prod(e.histsize)

function outcome_space(e::RectangularBinEncoding)
    # this is super simple :P could be optimized but its not a frequent operation
    return [decode(i, e) for i in 1:total_outcomes(e)]
end

##################################################################
# low level histogram call
##################################################################
# This method is called by `probabilities(x::Array_or_Dataset, est::ValueHistogram)`
"""
    fasthist(x::Vector_or_Dataset, ϵ::RectangularBinEncoding)
Intermediate method that runs `fasthist!` in the encoded space
and returns the encoded space histogram (counts) and corresponding bins.
"""
function fasthist(x, encoder::RectangularBinEncoding)
    bins = map(y -> encode(y, encoder), x)
    hist = fasthist!(bins)
    return hist, bins
end
