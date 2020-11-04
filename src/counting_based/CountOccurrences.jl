abstract type CountingBasedProbabilityEstimator <: ProbabilitiesEstimator end
export CountOccurrences, genentropy
import DelayEmbeddings: AbstractDataset

""" 
    CountOccurrences  <: CountingBasedProbabilityEstimator

A probabilities/entropy estimator based on straight-forward counting of distinct elements in 
a time series or multivariate dataset. From these counts, construct histograms. Sum-normalize
histograms to obtain probability distributions.
"""
struct CountOccurrences <: ProbabilitiesEstimator end


"""
# Entropy based on counting occurrences of distinct elements

    genentropy(x::AbstractDataset, est::CountOccurrences, α = 1; base = Base.MathConstants.e)
    genentropy(x::AbstractVector{T}, est::CountOccurrences, α = 1; base = Base.MathConstants.e) where T
    
Compute the order-`α` generalized (Rényi) entropy[^Rényi1960] of a multivariate dataset `x`
by counting repeated elements in `x`. Then, obtain a sum-normalized histogram from the 
counts of repeated elements, and compute generalized entropy.

Assume that `x` can be sorted.

If `x` is a `Dataset`, then identical state vectors are counted as repetitions. If `x`
is vector-like consisting of elements of the same type `T`, then identical elements of that 
type are counted as repetitions. 

## Example 

```julia
using Entropies, DelayEmbeddings

# A dataset with many identical state vectors
D = Dataset(rand(1:3, 5000, 3))

# Estimate order-1 generalized entropy to base 2 of the dataset
Entropies.genentropy(D, CountOccurrences(), 1, base = 2)
```

```julia
using Entropies, DelayEmbeddings

# A bunch of tuples, many potentially identical
x = [(rand(1:5), rand(1:5), rand(1:5)) for i = 1:10000]

# Estimate order-1 generalized entropy to base 2 of the tuples
Entropies.genentropy(x, CountOccurrences(), 1, base = 2)
```

See also: [`CountOccurrences`](@ref).
"""
function genentropy(x::AbstractDataset{m, T}, est::CountOccurrences, α::Real = 1; 
        base = Base.MathConstants.e) where {m, T <: Real}
    
    genentropy(α, non0hist(x, normalize = true), base = base)
end

function genentropy(x::AbstractVector{T}, est::CountOccurrences, α::Real = 1; 
        base = Base.MathConstants.e) where T
    
    genentropy(α, non0hist(x, normalize = true), base = base)
end

function probabilities(x::AbstractDataset, est::CountOccurrences; normalize::Bool = true)
    non0hist(x, est.binning, normalize = normalize)
end