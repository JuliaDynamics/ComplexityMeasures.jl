export RectangularBinEncoding
export RectangularBinning
export FixedRectangularBinning

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
const ValidFixedIntervalCount = Union{Integer, NTuple}
"""
    FixedRectangularBinning <: Encoding
    FixedRectangularBinning(ϵmin::E, ϵmax::E, N::Int) where E

Rectangular box partition of state space where the extent of the grid is explicitly
specified by `ϵmin` and `emax`, and along each dimension, the grid is subdivided into `N`
subintervals.

Binning instructions are deduced from the types of `ϵmin`/`emax` as follows:

1. `E<:Float64` sets the grid range along all dimensions to to `[ϵmin, ϵmax]`.
2. `E::NTuple{D, Float64}` sets ranges along each dimension individually, i.e.
    `[ϵmin[i], ϵmax[i]]` is the range along the `i`-th dimension.

If the grid spans the range `[r1, r2]` along a particular dimension, then this range
is subdivided into `N` subintervals of equal length `nextfloat((r2 - r1) / N)`.
Thus, for `m`-dimensional data, there are `N^m` boxes.
"""
struct FixedRectangularBinning{E, B} <: AbstractBinning
    ϵmin::E
    ϵmax::E
    N::B

    function FixedRectangularBinning(ϵmin::E1, ϵmax::E2, N::E3) where {E1 <: ValidFixedBinInputs, E2 <: ValidFixedBinInputs, E3<: ValidFixedIntervalCount}
        f_ϵmin = float.(ϵmin)
        f_ϵmax = float.(ϵmax)
        length(f_ϵmin) == length(f_ϵmax) || ErrorException("f_ϵmin and f_ϵmax have different lengths!")
        (length(N) == length(f_ϵmin) || length(N) == 1) || ErrorException("Bounds and subinterval counts are inconsistent!")
        return new{typeof(f_ϵmin)}(f_ϵmin, f_ϵmax, N)
    end
end

const Floating_or_Fixed_RectBinning = Union{RectangularBinning, FixedRectangularBinning}

# Internal function method extension for `probabilities`
function fasthist!(x::Vector_or_Dataset, ϵ::AbstractBinning)
    encoder = RectangularBinEncoding(x, ϵ)
    bins = outcomes(x, encoder)
    hist = fasthist!(bins)
    return Probabilities(hist), bins, encoder
end

"""
    RectangularBinEncoding <: Encoding
    RectangularBinEncoding(x, binning::RectangularBinning)
    RectangularBinEncoding(x, binning::FixedRectangularBinning)

Find the minima along each dimension, and compute appropriate
edge lengths for each dimension of `x` given a rectangular binning.
Put them in an `RectangularBinEncoding` that can be then used to map points into bins
via [`outcomes`](@ref).

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

function outcomes(x::Vector_or_Dataset, b::RectangularBinEncoding)
    return map(point -> encode_as_bin(point, b), x)
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


const RBE = RectangularBinEncoding
const RB = RectangularBinning
const NONDEDUCIBLE{T} = Union{
    RB{T},
    RBE{RB{T}}
    } where T <: Union{Q, Vector{Q}} where Q <: AbstractFloat

function total_outcomes(x, symbolization::NONDEDUCIBLE)
    msg = "total_outcomes can't be deduced from $NONDEDUCIBLE"
    throw(ArgumentError(msg))
end

# multiple-axis bins don't make sense for univariate input.
function total_outcomes(::AbstractVector,symbolization::RBE{RB{Vector{Int}}})
    msg = "total_outcomes is ambiguous for Vectors with RectangularBinning{Vector{Int}}"
    throw(ArgumentError(msg))
end
total_outcomes(::AbstractVector,symbolization::RBE{RB{Int}}) =
    symbolization.binning.ϵ
total_outcomes(::AbstractDataset{D}, symbolization::RBE{RB{Int}}) where {D} =
    symbolization.binning.ϵ^D
total_outcomes(::AbstractDataset{D}, symbolization::RBE{RB{Vector{Int}}}) where {D} =
    prod(symbolization.binning.ϵ)

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


# When the grid is fixed by the user, we can always deduce the total number of bins,
# even just from the binning itself - symbolization info not needed.
const FRB = FixedRectangularBinning
total_outcomes(b::FRB) = b.N
total_outcomes(::AbstractVector, b::FRB) = b.N
total_outcomes(::AbstractDataset{D, T}, b::FRB) where {D, T} = b.N^D
total_outcomes(symbolization::RBE{B, T}) where {B <: FRB, T <: Number} =
    symbolization.binning.N
total_outcomes(symbolization::RBE{B, T}) where {B <: FRB, T <: SVector{D}} where D =
    symbolization.binning.N^D
total_outcomes(x::AbstractVector, symbolization::RBE{B, T}) where {B <: FRB, T <: Number} =
    symbolization.binning.N
total_outcomes(x::AbstractDataset{D}, symbolization::RBE{B, T}) where {B <: FRB, T <: SVector{D}} where D =
    symbolization.binning.N^D
