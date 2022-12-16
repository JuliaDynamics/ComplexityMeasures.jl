export ValueHistogram, VisitationFrequency
# We need binning to be defined first to add it as a field to a struct
include("rectangular_binning.jl")
include("fasthist.jl")

"""
    ValueHistogram(x, b::AbstractBinning) <: ProbabilitiesEstimator
    ValueHistogram(b::FixedRectangularBinning) <: ProbabilitiesEstimator

A probability estimator based on binning the values of the data as dictated by
the binning scheme `b` and formally computing their histogram, i.e.,
the frequencies of points in the bins. An alias to this is `VisitationFrequency`.
Available binnings are:
- [`RectangularBinning`](@ref)
- [`FixedRectangularBinning`](@ref)

Notice that if not using the fixed binning, `x` (the input data) must also be given
to the estimator, as it is not possible to deduce histogram size only from the binning.

The `ValueHistogram` estimator has a linearithmic time complexity
(`n log(n)` for `n = length(x)`) and a linear space complexity (`l` for `l = dimension(x)`).
This allows computation of probabilities (histograms) of high-dimensional
datasets and with small box sizes `ε` without memory overflow and with maximum performance.
For performance reasons,
the probabilities returned never contain 0s and are arbitrarily ordered.

    ValueHistogram(x, ϵ::Union{Real,Vector})

A convenience method that accepts same input as [`RectangularBinning`](@ref)
and initializes this binning directly.

## Outcomes

The outcome space for `ValueHistogram` is the unique bins constructed
from `b`. Each bin is identified by its left (lowest-value) corner.
The bins are in data units, not integer (cartesian indices units), and
are returned as `SVector`s.

See also: [`RectangularBinning`](@ref).
"""
struct ValueHistogram{H<:HistogramEncoding} <: ProbabilitiesEstimator
    encoding::H
end
ValueHistogram(ϵ::Union{Real,Vector}) = ValueHistogram(RectangularBinning(ϵ))
function ValueHistogram(x, b::AbstractBinning)
    encoding = RectangularBinEncoding(x, b)
    return ValueHistogram(encoding)
end
function ValueHistogram(b::FixedRectangularBinning)
    encoding = RectangularBinEncoding(b)
    return ValueHistogram(encoding)
end


"""
    VisitationFrequency

An alias for [`ValueHistogram`](@ref).
"""
const VisitationFrequency = ValueHistogram

# The source code of `ValueHistogram` operates as rather simple calls to
# the underlying encoding and the `fasthist` function and extensions.
# See the `rectangular_binning.jl` file for more.
function probabilities(est::ValueHistogram, x)
    Probabilities(fasthist(x, est.encoding)[1])
end

function probabilities_and_outcomes(est::ValueHistogram, x)
    probs, bins = fasthist(x, est.encoding) # bins are integers here
    unique!(bins) # `bins` is already sorted from `fasthist!`
    # Here we transfor the cartesian coordinate based bins into data unit bins:
    outcomes = map(b -> decode(b, est.encoding), bins)
    return Probabilities(probs), vec(outcomes)
end

outcome_space(x, est::ValueHistogram) = outcome_space(est.encoding)
total_outcomes(x, est::ValueHistogram) = total_outcomes(est.encoding)
