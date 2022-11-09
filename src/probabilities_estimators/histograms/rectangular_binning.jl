# The binnings here are pretty much an intermediate interface to create a
# `RectangularBinEncoding`. Every actual source code extension is done on having
# an encoding...
export RectangularBinning, FixedRectangularBinning
export RectangularBinEncoding

##################################################################
# Structs and docstrings
##################################################################
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
struct RectangularBinEncoding{B, M, E} <: Encoding
    binning::B # either RectangularBinning or FixedRectangularBinning
    mini::M # fields are either static vectors or numbers
    edgelengths::E
end

function Base.show(io::IO, x::RectangularBinEncoding)
    return print(io, "RectangularBinEncoding\n" *
        "  binning: $(x.binning) \n" *
        "  box corners: $(x.mini)\n" *
        "  edgelengths: $(x.edgelengths)"
    )
end

function encode_as_bin(point, b::RectangularBinEncoding)
    (; mini, edgelengths) = b
    # Map a data point to its bin edge
    return floor.(Int, (point .- mini) ./ edgelengths)
end

function decode_from_bin(bin, b::RectangularBinEncoding)
    (; mini, edgelengths) = b
    # Remove one because we want lowest value corner, and we get indices starting from 1
    return (bin .- 1) .* edgelengths .+ mini
end

##################################################################
# Encoding bins using a *floating* (i.e. controlled by data) grid
##################################################################
function RectangularBinEncoding(x::AbstractDataset{D,T}, b::RectangularBinning;
        n_eps = 2) where {D, T}
    # This function always returns static vectors and is type stable
    ϵ = b.ϵ
    mini, maxi = minmaxima(x)
    v = ones(SVector{D,T})
    if ϵ isa Float64 || ϵ isa AbstractVector{<:AbstractFloat}
        edgelengths = ϵ .* v
    elseif ϵ isa Int || ϵ isa Vector{Int}
        edgeslengths_nonadjusted = @. (maxi - mini)/ϵ
        # Just taking nextfloat once here isn't enough for bins to cover data when using
        # `encode_as_bin` later, because subtraction and division leads to loss
        # of precision. We need a slightly bigger number, so apply nextfloat twice.
        edgelengths = nextfloat.(edgeslengths_nonadjusted, n_eps)
    else
        error("Invalid ϵ for binning of a dataset")
    end

    RectangularBinEncoding(b, mini, edgelengths)
end

function RectangularBinEncoding(x::AbstractVector{<:Real}, b::RectangularBinning; n_eps = 2)
    # This function always returns numbers and is type stable
    ϵ = b.ϵ
    mini, maxi = extrema(x)
    if ϵ isa AbstractFloat
        edgelength = ϵ
    elseif ϵ isa Int
        edgeslength_nonadjusted = (maxi - mini)/ϵ
        # Round-off occurs when encoding bins. Applying `nextfloat` twice seems to still
        # ensure that bins cover data. See comment above.
        edgelength = nextfloat(edgeslength_nonadjusted, n_eps)
    else
        error("Invalid ϵ for binning of a vector")
    end

    RectangularBinEncoding(b, mini, edgelength)
end

##################################################################
# Encoding bins using a fixed (user-specified) grid
##################################################################
function RectangularBinEncoding(::AbstractVector{<:Real},
        b::FixedRectangularBinning{E}; n_eps = 2) where E

    # This function always returns numbers and is type stable
    ϵmin, ϵmax = b.ϵmin, b.ϵmax
    mini = ϵmin
    if ϵmin isa Float64 && ϵmax isa Float64
        edgelength_nonadjusted = (ϵmax - ϵmin) / b.N
        edgelength = nextfloat(edgelength_nonadjusted, n_eps)
    else
        error("Invalid ϵmin or ϵmax for binning of a vector")
    end

    RectangularBinEncoding(b, mini, edgelength)
end

function RectangularBinEncoding(::AbstractDataset{D, T},
        b::FixedRectangularBinning{E}, n_eps = 2) where {D, T, E}
    # This function always returns static vectors and is type stable
    ϵmin, ϵmax = b.ϵmin, b.ϵmax
    if E <: Float64
        mini = SVector{D, Float64}(repeat([ϵmin], D))
        maxi = SVector{D, Float64}(repeat([ϵmax], D))
    elseif E <: NTuple{D}
        mini = SVector{D, Float64}(ϵmin)
        maxi = SVector{D, Float64}(ϵmax)
    else
        error("Invalid ϵmin or ϵmax for binning of a dataset")
    end

    edgelengths_nonadjusted = @. (maxi .- mini) / b.N
    edgelengths = nextfloat.(edgelengths_nonadjusted, n_eps)

    RectangularBinEncoding(b, mini, edgelengths)
end
# This version exists if the given `ϵ`s are already tuples.
# Then, the dataset doesn't need to be provided.
function RectangularBinEncoding(b::FixedRectangularBinning{<:NTuple}, n_eps = 2)
    ϵmin, ϵmax = b.ϵmin, b.ϵmax
    D = length(ϵmin)
    mini = SVector{D, Float64}(ϵmin)
    maxi = SVector{D, Float64}(ϵmax)
    edgelengths_nonadjusted = @. (maxi .- mini) / b.N
    edgelengths = nextfloat.(edgelengths_nonadjusted, n_eps)
    RectangularBinEncoding(b, mini, edgelengths)
end

##################################################################
# Outcomes / total outcomes / extension of functions
##################################################################
# When the grid is fixed by the user, we can always deduce the total number of bins,
# even just from the binning itself - symbolization info not needed.
function total_outcomes(e::RectangularBinEncoding)
    if e.binning isa RectangularBinning
        error("Not possible to _uniquely_ define for `RectangularBinning`.")
    end
    D = length(e.mini)
    return e.binning.N^D
end

# This function does not need `x`; all info about binning are in the encoding
function outcome_space(e::RectangularBinEncoding)
    if e.binning isa RectangularBinning
        error("Not possible to _uniquely_ define for `RectangularBinning`.")
    end
    # We can be smart here. All possible bins are exactly the same thing
    # as the Cartesian Indices of an array, mapped into "data" units
    dims = _array_dims_from_fixed_binning(e)
    bins = CartesianIndices(dims)
    outcomes = map(b -> decode_from_bin(b, e), bins)
    return outcomes
end

function _array_dims_from_fixed_binning(e)
    N = e.binning.N
    D = length(e.mini)
    return ntuple(i -> N, D)
end

##################################################################
# low level histogram call
##################################################################
# This method is called by `probabilities(x::Array_or_Dataset, est::ValueHistogram)`
"""
    fasthist(x::Vector_or_Dataset, ϵ::AbstractBinning)
Create an encoding for binning, then map `x` to bins, then call `fasthist!` on the bins.
Return the output probabilities, the bins, and the created encoder.
"""
function fasthist(x::Vector_or_Dataset, ϵ::AbstractBinning)
    encoder = RectangularBinEncoding(x, ϵ)
    bins = outcomes(x, encoder)
    hist = fasthist!(bins)
    return Probabilities(hist), bins, encoder
end
