using DelayEmbeddings
import DelayEmbeddings: AbstractDataset, Dataset
export ProbabilitiesEstimator, Probabilities
export probabilities, probabilities!, entropy

const Vector_or_Dataset = Union{AbstractVector, Dataset}

"""
    Probabilities(x) → p
A simple wrapper type around an `x::AbstractVector` which ensures that `p` sums to 1.
Behaves identically to `Vector`.
"""
struct Probabilities{T}
    p::Vector{T}
    function Probabilities(x::AbstractVector)
        T = eltype(x)
        s = sum(x)
        if s ≠ 1
            x = x ./ s
        end
        return new{T}(x)
    end
end

# extend base Vector interface:
for f in (:length, :size, :IteratorSize, :eachindex, :eltype, :lastindex, :firstindex)
    @eval Base.$(f)(d::Probabilities) = $(f)(d.p)
end
@inline Base.iterate(d::Probabilities, i = 1) = iterate(d.p, i)
@inline Base.getindex(d::Probabilities, i) = d.p[i]
@inline Base.:*(d::Probabilities, x::Number) = d.p * x
@inline Base.sum(d::Probabilities{T}) = one(T)

"""
An abstract type for probabilities estimators.
"""
abstract type ProbabilitiesEstimator end
const ProbEst = ProbabilitiesEstimator # shorthand

"""
    probabilities(x::Dataset, est::ProbabilitiesEstimator) → p::Probabilities
Calculate probabilities representing the given dataset `x` based on the provided
estimator and return them as a [`Probabilities`](@ref) container (`Vector`-like).
The probabilities are typically unordered and may or may not contain 0s, see the
documentation of the individual estimators for more.

The configuration options are always given as arguments to the chosen estimator.

    probabilities(x::Dataset, ε::Real) → p::Probabilities
Convenience syntax which provides probabilities for `x` based on rectangular binning.
In short, the state space is divided into boxes of length `ε`, see
[`RectangularBinning`](@ref) for more.

    probabilities(x::Dataset_or_Vector) → p::Probabilities
Directly count probabilities from the elements of `x` without any discretization,
binning, or other processing (mostly useful when `x` contains categorical or integer data).
"""
function probabilities end

"""
    probabilities!(p, args...)
Identical with `probabilities(args...)` but writes the estimated probabilities in the
pre-allocated vector `p` instead of making a new one.
"""
function probabilities! end

"""
    entropy(p::Probabilities, α = 1.0; base = Base.MathConstants.e)

Compute the generalized order-`α` entropy of some probabilities
returned by the [`probabilities`](@ref) function.

    entropy(x::Vector_or_Dataset, est::ProbabilityEstimator, α = 1.0; base)

A convenience syntax, which calls first `probabilities(x, est)`
and then calculates the entropy of the result.

## Description

Let ``p`` be an array of probabilities (summing to 1). Then the generalized (Rényi) entropy is

```math
H_\\alpha(p) = \\frac{1}{1-\\alpha} \\log \\left(\\sum_i p[i]^\\alpha\\right)
```

and generalizes other known entropies,
like e.g. the information entropy
(``\\alpha = 1``, see [^Shannon1948]), the maximum entropy (``\\alpha=0``,
also known as Hartley entropy), or the correlation entropy
(``\\alpha = 2``, also known as collision entropy).

[^Rényi1960]: A. Rényi, *Proceedings of the fourth Berkeley Symposium on Mathematics, Statistics and Probability*, pp 547 (1960)
[^Shannon1948]: C. E. Shannon, Bell Systems Technical Journal **27**, pp 379 (1948)
"""
function entropy(p::Probabilities, α = 1.0; base = Base.MathConstants.e)
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

function entropy(x, est::ProbEst, α = 1.0; base = Base.MathConstants.e)
    p = probabilities(x, est)
    entropy(p, α; base)
end
