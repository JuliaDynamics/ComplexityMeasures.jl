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

"""
# Probabilities based on binning (visitation frequency)

    probabilities(x::AbstractDataset, est::VisitationFrequency) → ps::Probabilities

Superimpose a rectangular grid (bins/boxes) dictated by `est` over the data `x` and return
the sum-normalized histogram (i.e. frequency at which the points of `x` visits the bins/boxes
in the grid) in an unordered 1D form, discarding all non-visited bins and bin edge information.

# Performances Notes

This method has a linearithmic time complexity `(n log(n)` for `n = length(data)`) and a
linear space complexity `l` for `l = dimension(data)`). This allows computation of
histograms of high-dimensional datasets and with small box sizes ε without memory
overflow and with maximum performance.

See also: [`VisitationFrequency`](@ref), [`RectangularBinning`](@ref).

# Example

```julia
using Entropies, DelayEmbeddings
D = Dataset(rand(100, 3))

# How shall the data be partitioned?
# Here, we subdivide each coordinate axis into 4 equal pieces
# over the range of the data, resulting in rectangular boxes/bins
ϵ = RectangularBinning(4)

# Feed partitioning instructions to estimator.
est = VisitationFrequency(ϵ)

# Estimate a probability distribution over the partition
probabilities(D, est)
```
"""
function probabilities(x::AbstractDataset, est::VisitationFrequency)
    _non0hist(x, est.binning)[1]
end
