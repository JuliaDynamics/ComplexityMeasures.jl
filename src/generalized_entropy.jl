export genentropy
import DelayEmbeddings: AbstractDataset

"""
    genentropy(α::Real, p::AbstractArray; base = Base.MathConstants.e)

Compute the entropy, to the given `base`, of an array of probabilities `p` (assuming 
that `p` is sum-normalized).

If a multivariate `Dataset` `x` is given, then the a sum-normalized histogram is obtained 
directly on the elements of `x`, and the generalized entropy is computed on that 
distribution.

## Description

Let ``p`` be an array of probabilities (summing to 1). Then the Rényi entropy is

```math
H_\\alpha(p) = \\frac{1}{1-\\alpha} \\log \\left(\\sum_i p[i]^\\alpha\\right)
```

and generalizes other known entropies,
like e.g. the information entropy
(``\\alpha = 1``, see [^Shannon1948]), the maximum entropy (``\\alpha=0``,
also known as Hartley entropy), or the correlation entropy
(``\\alpha = 2``, also known as collision entropy).

## Example 

```julia
using Entropies
p = rand(5000)
p = p ./ sum(p) # normalizing to 1 ensures we have a probability distribution

# Estimate order-1 generalized entropy to base 2 of the distribution
Entropies.genentropy(1, ps, base = 2)
```

See also: [`non0hist`](@ref).

[^Rényi1960]: A. Rényi, *Proceedings of the fourth Berkeley Symposium on Mathematics, Statistics and Probability*, pp 547 (1960)
[^Shannon1948]: C. E. Shannon, Bell Systems Technical Journal **27**, pp 379 (1948)
"""
function genentropy(α::Real, p::AbstractArray{T}; base = Base.MathConstants.e) where {T <: Real}
    α < 0 && throw(ArgumentError("Order of generalized entropy must be ≥ 0."))
    if α ≈ 0
        return log(base, length(p)) #Hartley entropy, max-entropy
    elseif α ≈ 1
        return -sum( x*log(base, x) for x in p ) #Shannon entropy
    elseif isinf(α)
        return -log(base, maximum(p)) #Min entropy
    else
        return (1/(1-α))*log(base, sum(x^α for x in p) ) #Renyi α entropy
    end
end
