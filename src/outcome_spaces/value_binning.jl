export ValueBinning, ValueHistogram, VisitationFrequency
# Binnings are defined in the encoding folder!

"""
    ValueBinning(b::AbstractBinning) <: OutcomeSpace

An [`OutcomeSpace`](@ref) based on binning the values of the data as dictated by
the binning scheme `b` and formally computing their histogram, i.e.,
the frequencies of points in the bins. An alias to this is `VisitationFrequency`.
Available binnings are subtypes of [`AbstractBinning`](@ref).

The `ValueBinning` estimator has a linearithmic time complexity
(`n log(n)` for `n = length(x)`) and a linear space complexity (`l` for `l = dimension(x)`).
This allows computation of probabilities (histograms) of high-dimensional
datasets and with small box sizes `ε` without memory overflow and with maximum performance.
For performance reasons,
the probabilities returned never contain 0s and are arbitrarily ordered.

    ValueBinning(ϵ::Union{Real,Vector})

A convenience method that accepts same input as [`RectangularBinning`](@ref)
and initializes this binning directly.

## Outcomes

The outcome space for `ValueBinning` is the unique bins constructed
from `b`. Each bin is identified by its left (lowest-value) corner,
because bins are always left-closed-right-open intervals `[a, b)`.
The bins are in data units, not integer (cartesian indices units), and
are returned as `SVector`s, i.e., same type as input data.

For convenience, [`outcome_space`](@ref)
returns the outcomes in the same array format as the underlying binning
(e.g., `Matrix` for 2D input).

For [`FixedRectangularBinning`](@ref) the [`outcome_space`](@ref) is well-defined from the
binning, but for [`RectangularBinning`](@ref) input `x` is needed as well.

## Implements

- [`codify`](@ref). Used for encoding inputs where ordering matters (e.g. time series).
"""
struct ValueBinning{B<:AbstractBinning} <: CountBasedOutcomeSpace
    binning::B
end
ValueBinning(ϵ::Union{Real,Vector}) = ValueBinning(RectangularBinning(ϵ))

"""
    VisitationFrequency

An alias for [`ValueBinning`](@ref).
"""
const VisitationFrequency = ValueBinning

"""
    ValueHistogram

An alias for [`ValueBinning`](@ref).
"""
const ValueHistogram = ValueBinning

# --------------------------------------------------------------------------------
# The source code of `ValueBinning` operates as rather simple calls to
# the underlying encoding and the `fasthist`/`fasthist!` functions.
# See the `encoding_implementations/rectangular_binning.jl` file for more.
# --------------------------------------------------------------------------------
# Explicitly override `counts` here, because it is more efficient to not
# decode the outcomes.
function counts(est::ValueBinning, x)
    return counts(RectangularBinEncoding(est.binning, x), x)
end

function counts_and_outcomes(est::ValueBinning, x)
    return counts_and_outcomes(RectangularBinEncoding(est.binning, x), x)
end

# Explicitly override `probabilities` here, because it is more efficient to not
# decode the outcomes.
function probabilities(est::ValueBinning, x)
    return Probabilities(counts(RectangularBinEncoding(est.binning, x), x))
end

function probabilities_and_outcomes(est::ValueBinning, x)
    cts, outs = counts_and_outcomes(RectangularBinEncoding(est.binning, x), x)
    probs = Probabilities(cts, outs)
    return probs, outcomes(probs)
end

function outcome_space(est::ValueBinning, x)
    return outcome_space(RectangularBinEncoding(est.binning, x))
end

function outcome_space(est::ValueBinning{<:FixedRectangularBinning})
    return outcome_space(RectangularBinEncoding(est.binning))
end

function codify(o::ValueBinning{<:FixedRectangularBinning{D}}, x::AbstractVector) where D
    verify_input(o.binning, x)
    encoder = RectangularBinEncoding(o.binning)
    # TODO: should we warn if points outside the binning are considered? Probably not,
    # since being outside the binning is a valid state.
    return encode.(Ref(encoder), x)
end

function codify(o::ValueBinning{<:FixedRectangularBinning}, x::AbstractStateSpaceSet{D}) where D
    verify_input(o.binning, x)
    encoder = RectangularBinEncoding(o.binning)
    return encode.(Ref(encoder), x.data)
end

function codify(o::ValueBinning{<:RectangularBinning}, x::AbstractVector{<:Real})
    encoder = RectangularBinEncoding(o.binning, x)
    return encode.(Ref(encoder), x)
end

function codify(o::ValueBinning{<:RectangularBinning}, x::AbstractStateSpaceSet{D}) where D
    encoder = RectangularBinEncoding(o.binning, x)
    return encode.(Ref(encoder), x.data)
end

# Some input checks
#----------------------------------------------------------------
function verify_input(f::FixedRectangularBinning, x::AbstractStateSpaceSet{D}) where D
    if length(f.ranges) != D
        l = length(f.ranges)
        s = "The number of ranges for the `FixedRectangularBinning` is $l, but the input"*
            " `StateSpaceSet` is $D-dimensional. Please provide a "*
                "`FixedRectangularBinning` with $D ranges."
        throw(DimensionMismatch(s))
    end
end
function verify_input(f::FixedRectangularBinning, x::AbstractVector)
    if length(f.ranges) != 1
        l = length(f.ranges)
        s = "The number of ranges for the `FixedRectangularBinning` is $l, but the "*
            " dimension is 1-dimensional (a vector). Please provide a "*
                "`FixedRectangularBinning` with only 1 range."
        throw(DimensionMismatch(s))
    end
end
