
"""
    genentropy(p::Probabilities; q = 1.0, base = MathConstants.e)

Compute the generalized order-`q` entropy of some probabilities
returned by the [`probabilities`](@ref) function. Alternatively, compute entropy
from pre-computed `Probabilities`.

    genentropy(x::Array_or_Dataset, est; q = 1.0, base)

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

[^Rényi1960]: A. Rényi, *Proceedings of the fourth Berkeley Symposium on Mathematics,
    Statistics and Probability*, pp 547 (1960)
[^Shannon1948]: C. E. Shannon, Bell Systems Technical Journal **27**, pp 379 (1948)
"""
function genentropy end

function genentropy(prob::Probabilities; q = 1.0, α = nothing, base = MathConstants.e)
    Base.depwarn("`genentropy(x)` is deprecated, use `renyientropy(x)` instead.", :genentropy)

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

function genentropy(::AbstractArray{<:Real})
    Base.depwarn("`genentropy(x)` is deprecated, use `renyientropy(x)` instead.", :genentropy)
    error("For single-argument input, do `genentropy(Probabilities(x))` instead.")
end

function genentropy(x::Array_or_Dataset, est; q = 1.0, α = nothing, base = MathConstants.e)
    Base.depwarn("`genentropy(x, est)` is deprecated, use `renyientropy(x, est)` instead.", :genentropy)
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
    Base.depwarn("`genentropy(p, x, est)` is deprecated, use `renyientropy(p, x, est)` instead.", :genentropy)

    if α ≠ nothing
        @warn "Keyword `α` is deprecated in favor of `q`."
        q = α
    end
    probabilities!(p, x, est)
    genentropy(p; q = q, base = base)
end
