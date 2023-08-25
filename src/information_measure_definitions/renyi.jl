export Renyi

"""
    Renyi <: InformationMeasure
    Renyi(q, base = 2)
    Renyi(; q = 1.0, base = 2)

The Rényi generalized order-`q` entropy [Rényi1961](@cite), used with [`information`](@ref)
to compute an entropy with units given by `base` (typically `2` or `MathConstants.e`).

## Description

Let ``p`` be an array of probabilities (summing to 1).
Then the Rényi generalized entropy is

```math
H_q(p) = \\frac{1}{1-q} \\log \\left(\\sum_i p[i]^q\\right)
```

and generalizes other known entropies,
like e.g. the information entropy
(``q = 1``, see [Shannon1948](@citet)), the maximum entropy (``q=0``,
also known as Hartley entropy), or the correlation entropy
(``q = 2``, also known as collision entropy).

The maximum value of the Rényi entropy is ``\\log_{base}(L)``, which is the entropy of the
uniform distribution with ``L`` the [`total_outcomes`](@ref).
"""
Base.@kwdef struct Renyi{Q, B} <: Entropy
    q::Q = 1.0
    base::B  = 2.0
end
Renyi(q) = Renyi(q, 2)

function information(e::Renyi, probs::Probabilities)
    q, base = e.q, e.base
    q < 0 && throw(ArgumentError("Order of Renyi entropy must be ≥ 0."))
    non0_probs = Iterators.filter(!iszero, vec(probs))
    logf = log_with_base(base)
    if q ≈ 0
        return logf(count(!iszero, probs))
    elseif q ≈ 1
        return -sum(x*logf(x) for x in non0_probs)
    elseif isinf(q)
        return -logf(maximum(non0_probs))
    else
        return (1/(1-q))*logf(sum(x^q for x in non0_probs))
    end
end

information_maximum(e::Renyi, L::Int) = log_with_base(e.base)(L)
