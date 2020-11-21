export VisitationFrequency, probabilities
import DelayEmbeddings: Dataset, AbstractDataset

"""
    VisitationFrequency(r::RectangularBinning) <: BinningProbabilitiesEstimator

A probability estimator based on binning data into rectangular boxes dictated by
the binning scheme `r`.


## Example

```julia
# Construct boxes by dividing each coordinate axis into 5 equal-length chunks.
b = RectangularBinning(5)

# A probabilities estimator that, when applied a dataset, computes visitation frequencies
# over the boxes of the binning, constructed as describedon the previous line.
est = VisitationFrequency(b)
```

See also: [`RectangularBinning`](@ref).
"""
struct VisitationFrequency <: BinningProbabilitiesEstimator
    binning::RectangularBinning

    function VisitationFrequency(r::RectangularBinning)
        new(r)
    end
end

function probabilities(x::AbstractDataset, est::VisitationFrequency)
    _non0hist(x, est.binning)[1]
end
