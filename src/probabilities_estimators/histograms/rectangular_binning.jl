export RectangularBinning, FixedRectangularBinning
export RectangularBinEncoding

abstract type AbstractBinning end
abstract type HistogramEncoding <: Encoding end

##################################################################
# Structs and docstrings
##################################################################
# Notice that the binning types are intermediate structs that are not used directly
# in the source code. Their only purpose is instructions of how to create a
# `RectangularBinEncoder`. All actual source code functionality of `ValueHistogram`
# is implemented based on `RectangularBinEncoder`.

"""
    RectangularBinning(ϵ) <: AbstractBinning

Rectangular box partition of state space using the scheme `ϵ`,
deducing the coordinates of the grid axis minima from the input data.
Generally it is preferred to use [`FixedRectangularBinning`](@ref) instead,
as it has a well defined outcome space without knowledge of input data.

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
    FixedRectangularBinning(ϵmin::NTuple, ϵmax::NTuple, N::Int)

Rectangular box partition of state space where the extent of the grid is explicitly
specified by `ϵmin` and `emax`, and along each dimension, the grid is subdivided into `N`
subintervals. Points falling outside the partition do not attribute to probabilities.
This binning type leads to a well-defined outcome space without knowledge of input,
see [`ValueHistogram`](@ref).

`ϵmin`/`emax` must be `NTuple{D, <:Real}` for input of `D`-dimensional data.

    FixedRectangularBinning(ϵmin::Real, ϵmax::Real, N::Int, D::Int = 1)

This is a convenience method where each dimension of the binning has the same extent
and the input data are `D` dimensional, which defaults to 1 (timeseries).
"""
struct FixedRectangularBinning{D,T<:Real} <: AbstractBinning
    ϵmin::NTuple{D,T}
    ϵmax::NTuple{D,T}
    N::Int
end
function FixedRectangularBinning(ϵmin::Real, ϵmax::Real, N::Int, D::Int = 1)
    FixedRectangularBinning(ntuple(x->ϵmin, D), ntuple(x->ϵmax, D), N)
end

"""
    RectangularBinEncoding <: Encoding
    RectangularBinEncoding(binning::RectangularBinning, x; n_eps = 2)
    RectangularBinEncoding(binning::FixedRectangularBinning; n_eps = 2)

Struct used in [`outcomes`](@ref) to map points of `x` into their respective bins.
It finds the minima along each dimension, and computes appropriate
edge lengths for each dimension of `x` given a rectangular binning.

The second signature does not need `x` because (1) the binning is fixed, and the
size of `x` doesn't matter, and (2) because the binning contains the dimensionality
information as `ϵmin/max` is already an `NTuple`.

Due to roundoff error when computing bin edges, a small tolerance `n_eps` of `n_eps * eps()`
is added to bin widths to ensure the correct number of bins is produced.

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
        "  box corners: $(x.mini)\n" *
        "  edgelengths: $(x.edgelengths)\n" *
        "  histogram size: $(x.histsize)"
    )
end

function encode(e::RectangularBinEncoding, point)
    (; mini, edgelengths) = e
    # Map a data point to its bin edge (plus one because indexing starts from 1)
    bin = floor.(Int, (point .- mini) ./ edgelengths) .+ 1
    cartidx = CartesianIndex(Tuple(bin))
    # We have decided on the arbitrary convention that out of bound points
    # will get the special symbol `-1`. Erroring doesn't make sense as it is expected
    # that for fixed histograms there may be points outside of them.
    if checkbounds(Bool, e.li, cartidx)
        return @inbounds e.li[cartidx]
    else
        return -1
    end
end

function decode(e::RectangularBinEncoding{B, D, T}, bin::Int) where {B, D, T}
    V = SVector{D,T}
    if checkbounds(Bool, e.ci, bin)
        @inbounds cartesian = e.ci[bin]
    else
        error("Cannot decode integer $(bin): out of bounds of underlying histogram.")
    end
    (; mini, edgelengths) = e
    # Remove one because we want lowest value corner, and we get indices starting from 1
    return (V(Tuple(cartesian)) .- 1) .* edgelengths .+ mini
end

##################################################################
# Initialization of encodings
##################################################################
# Data-controlled grid
function RectangularBinEncoding(b::RectangularBinning, x; n_eps = 2)
    # This function always returns static vectors and is type stable
    D = dimension(x)
    T = eltype(x)
    ϵ = b.ϵ
    mini, maxi = minmaxima(x)
    v = ones(SVector{D,T})
    if ϵ isa Float64 || ϵ isa AbstractVector{<:AbstractFloat}
        edgelengths = SVector{D,T}(ϵ .* v)
        histsize = ceil.(Int, nextfloat.((maxi .- mini), n_eps) ./ edgelengths)
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
function RectangularBinEncoding(b::FixedRectangularBinning{D, T}; n_eps = 2) where {D,T}
    # This function always returns static vectors and is type stable
    ϵmin, ϵmax = b.ϵmin, b.ϵmax
    mini = SVector{D, T}(ϵmin)
    maxi = SVector{D, T}(ϵmax)
    edgelengths_nonadjusted = @. (maxi .- mini) / b.N
    edgelengths = nextfloat.(edgelengths_nonadjusted, n_eps)
    histsize = SVector{D,Int}(fill(b.N, D))
    ci = CartesianIndices(Tuple(histsize))
    li = LinearIndices(ci)
    RectangularBinEncoding(b, typeof(edgelengths)(mini), edgelengths, histsize, ci, li)
end

##################################################################
# Outcomes / total outcomes
##################################################################
total_outcomes(e::RectangularBinEncoding) = prod(e.histsize)

function outcome_space(e::RectangularBinEncoding)
    # this is super simple :P could be optimized but its not a frequent operation
    return [decode(e, i) for i in 1:total_outcomes(e)]
end

##################################################################
# low level histogram call
##################################################################
# This method is called by `probabilities(est::ValueHistogram, x::Array_or_Dataset)`
"""
    fasthist(c::RectangularBinEncoding, x::Vector_or_Dataset)
Intermediate method that runs `fasthist!` in the encoded space
and returns the encoded space histogram (counts) and corresponding bins.
Also skips any instances of out-of-bound points for the histogram.
"""
function fasthist(encoder::RectangularBinEncoding, x)
    bins = map(y -> encode(encoder, y), x)
    # We discard `-1`, as it encodes points outside the histogram limit
    # (which should only happen for `Fixed` binnings)
    discard_minus_ones!(bins)
    hist = fasthist!(bins)
    return hist, bins
end

function discard_minus_ones!(bins)
    idxs = findall(isequal(-1), bins)
    deleteat!(bins, idxs)
end
