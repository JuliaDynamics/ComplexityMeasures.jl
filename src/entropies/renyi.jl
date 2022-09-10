export entropy_renyi

"""
    entropy_renyi(p::Probabilities; q = 1.0, base = MathConstants.e)

Compute the Rényi[^Rényi1960] generalized order-`q` entropy of some probabilities
(typically returned by the [`probabilities`](@ref) function).

    entropy_renyi(x::Array_or_Dataset, est; q = 1.0, base)

A convenience syntax, which calls first `probabilities(x, est)`
and then calculates the entropy of the result (and thus `est` can be anything
the [`probabilities`](@ref) function accepts).

## Description

Let ``p`` be an array of probabilities (summing to 1).
Then the Rényi generalized entropy is

```math
H_q(p) = \\frac{1}{1-q} \\log \\left(\\sum_i p[i]^q\\right)
```

and generalizes other known entropies,
like e.g. the information entropy
(``q = 1``, see [^Shannon1948]), the maximum entropy (``q=0``,
also known as Hartley entropy), or the correlation entropy
(``q = 2``, also known as collision entropy).

[^Rényi1960]:
    A. Rényi, _Proceedings of the fourth Berkeley Symposium on Mathematics,
    Statistics and Probability_, pp 547 (1960)
[^Shannon1948]: C. E. Shannon, Bell Systems Technical Journal **27**, pp 379 (1948)
"""
function entropy_renyi end

function entropy_renyi(prob::Probabilities; q = 1.0, α = nothing, base = MathConstants.e)
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

entropy_renyi(::AbstractArray{<:Real}) =
    error("For single-argument input, do `entropy_renyi(Probabilities(x))` instead.")

function entropy_renyi(x::Array_or_Dataset, est; q = 1.0, α = nothing, base = MathConstants.e)
    if α ≠ nothing
        @warn "Keyword `α` is deprecated in favor of `q`."
        q = α
    end
    p = probabilities(x, est)
    entropy_renyi(p; q = q, base = base)
end

"""
    entropy_renyi!(p, x, est::ProbabilitiesEstimator; q = 1.0, base = MathConstants.e)

Similarly with `probabilities!` this is an in-place version of `entropy_renyi` that allows
pre-allocation of temporarily used containers.

Only works for certain estimators. See for example [`SymbolicPermutation`](@ref).
"""
function entropy_renyi!(p, x, est; q = 1.0, α = nothing, base = MathConstants.e)
    if α ≠ nothing
        @warn "Keyword `α` is deprecated in favor of `q`."
        q = α
    end
    probabilities!(p, x, est)
    entropy_renyi(p; q = q, base = base)
end
