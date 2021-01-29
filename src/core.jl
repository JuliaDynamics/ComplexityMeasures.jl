using DelayEmbeddings
import DelayEmbeddings: AbstractDataset, Dataset, dimension
export ProbabilitiesEstimator, Probabilities
export EntropyEstimator
export probabilities, probabilities!
export genentropy, genentropy!
export Dataset, dimension

const Vector_or_Dataset = Union{AbstractVector, AbstractDataset}

"""
    Probabilities(x) → p
A simple wrapper type around an `x::AbstractVector` which ensures that `p` sums to 1.
Behaves identically to `Vector`.
"""
struct Probabilities{T} <: AbstractVector{T}
    p::Vector{T}
    function Probabilities(x::AbstractVector{T}) where T
        s = sum(x)
        if s ≠ 1
            x = x ./ s
        end
        return new{T}(x)
    end
end

# extend base Vector interface:
for f in (:length, :size, :eachindex, :eltype, :lastindex, :firstindex)
    @eval Base.$(f)(d::Probabilities) = $(f)(d.p)
end
Base.IteratorSize(d::Probabilities) = Base.HasLength()
@inline Base.iterate(d::Probabilities, i = 1) = iterate(d.p, i)
@inline Base.getindex(d::Probabilities, i) = d.p[i]
@inline Base.setindex!(d::Probabilities, v, i) = (d.p[i] = v)
@inline Base.:*(d::Probabilities, x::Number) = d.p * x
@inline Base.sum(d::Probabilities{T}) where T = one(T)

"""
An abstract type for entropy estimators that don't explicitly estimate probabilities,
but return the value of the entropy directly.
"""
abstract type EntropyEstimator end
const EntEst = EntropyEstimator # shorthand

"""
An abstract type for probabilities estimators.
"""
abstract type ProbabilitiesEstimator end
const ProbEst = ProbabilitiesEstimator # shorthand

"""
    probabilities(x::Vector_or_Dataset, est::ProbabilitiesEstimator) → p::Probabilities

Calculate probabilities representing `x` based on the provided
estimator and return them as a [`Probabilities`](@ref) container (`Vector`-like).
The probabilities are typically unordered and may or may not contain 0s, see the
documentation of the individual estimators for more.

The configuration options are always given as arguments to the chosen estimator.

    probabilities(x::Vector_or_Dataset, ε::AbstractFloat) → p::Probabilities

Convenience syntax which provides probabilities for `x` based on rectangular binning
(i.e. performing a histogram). In short, the state space is divided into boxes of length
`ε`, and formally we use `est = VisitationFrequency(RectangularBinning(ε))`
as an estimator, see [`VisitationFrequency`](@ref).

This method has a linearithmic time complexity (`n log(n)` for `n = length(x)`)
and a linear space complexity (`l` for `l = dimension(x)`).
This allows computation of probabilities (histograms) of high-dimensional
datasets and with small box sizes `ε` without memory overflow and with maximum performance.
To obtain the bin information along with `p`, use [`binhist`](@ref).

    probabilities(x::Vector_or_Dataset, n::Integer) → p::Probabilities
Same as the above method, but now each dimension of the data is binned into `n::Int` equal
sized bins instead of bins of length `ε::AbstractFloat`.

    probabilities(x::Vector_or_Dataset) → p::Probabilities
Directly count probabilities from the elements of `x` without any discretization,
binning, or other processing (mostly useful when `x` contains categorical or integer data).
"""
function probabilities end

"""
    probabilities!(args...)
Identical to `probabilities(args...)`, but allows pre-allocation of temporarily used
containers.

Only works for certain estimators. See for example [`SymbolicPermutation`](@ref).
"""
function probabilities! end

"""
    genentropy(p::Probabilities; q = 1.0, base = MathConstants.e)

Compute the generalized order-`q` entropy of some probabilities
returned by the [`probabilities`](@ref) function. Alternatively, compute entropy
from pre-computed `Probabilities`.

    genentropy(x::Vector_or_Dataset, est; q = 1.0, base)

A convenience syntax, which calls first `probabilities(x, est)`
and then calculates the entropy of the result (and thus `est` can be a
`ProbabilitiesEstimator` or simply `ε::Real`).

## Description

Let ``p`` be an array of probabilities (summing to 1). Then the generalized (Rényi) entropy is

```math
H_q(p) = \\frac{1}{1-q} \\log \\left(\\sum_i p[i]^q\\right)
```

and generalizes other known entropies,
like e.g. the information entropy
(``q = 1``, see [^Shannon1948]), the maximum entropy (``q=0``,
also known as Hartley entropy), or the correlation entropy
(``q = 2``, also known as collision entropy).

[^Rényi1960]: A. Rényi, *Proceedings of the fourth Berkeley Symposium on Mathematics, Statistics and Probability*, pp 547 (1960)
[^Shannon1948]: C. E. Shannon, Bell Systems Technical Journal **27**, pp 379 (1948)
"""
function genentropy end

function genentropy(prob::Probabilities; q = 1.0, α = nothing, base = MathConstants.e)
    if α ≠ nothing
        @warn "Keyword `α` is deprecated in favor of `q`."
        q = α
    end
    q < 0 && throw(ArgumentError("Order of generalized entropy must be ≥ 0."))
    haszero = any(iszero, prob)
    p = if haszero
        i0 = findall(iszero, prob.p)
        # We copy because if someone initialized Probabilities with 0s, I would guess
        # they would want to index the zeros as well. Not so costly anyways.
        deleteat!(copy(prob.p), i0)
    else
        prob.p
    end

    if q ≈ 0
        return log(base, length(p)) #Hartley entropy, max-entropy
    elseif q ≈ 1
        return -sum( x*log(base, x) for x in p ) #Shannon entropy
    elseif isinf(q)
        return -log(base, maximum(p)) #Min entropy
    else
        return (1/(1-q))*log(base, sum(x^q for x in p) ) #Renyi q entropy
    end
end

genentropy(x::AbstractArray{<:Real}) =
    error("For single-argument input, do `genentropy(Probabilities(x))` instead.")

function genentropy(x::Vector_or_Dataset, est; q = 1.0, α = nothing, base = MathConstants.e)
    if α ≠ nothing
        @warn "Keyword `α` is deprecated in favor of `q`."
        q = α
    end
    p = probabilities(x, est)
    genentropy(p; q = q, base = base)
end

"""
    genentropy!(p, x, est::ProbabilitiesEstimator; q = 1.0, base = MathConstants.e)

Similarly with `probabilities!` this is an in-place version of `genentropy` that allows
pre-allocation of temporarily used containers.

Only works for certain estimators. See for example [`SymbolicPermutation`](@ref).
"""
function genentropy!(p, x, est; q = 1.0, α = nothing, base = MathConstants.e)
    if α ≠ nothing
        @warn "Keyword `α` is deprecated in favor of `q`."
        q = α
    end
    probabilities!(p, x, est)
    genentropy(p; q = q, base = base)
end
