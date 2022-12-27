export Renyi, Shannon

"""
    Renyi <: EntropyDefinition
    Renyi(q, base = 2)
    Renyi(; q = 1.0, base = 2)

The Rényi[^Rényi1960] generalized order-`q` entropy, used with [`entropy`](@ref)
to compute an entropy with units given by `base` (typically `2` or `MathConstants.e`).

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

The maximum value of the Rényi entropy is ``\\log_{base}(L)``, which is the entropy of the
uniform distribution with ``L`` the [`total_outcomes`](@ref).

[^Rényi1960]:
    A. Rényi, _Proceedings of the fourth Berkeley Symposium on Mathematics,
    Statistics and Probability_, pp 547 (1960)

[^Shannon1948]: C. E. Shannon, Bell Systems Technical Journal **27**, pp 379 (1948)
"""
Base.@kwdef struct Renyi{Q, B} <: EntropyDefinition
    q::Q = 1.0
    base::B  = 2.0
end
Renyi(q) = Renyi(q, 2)

function entropy(e::Renyi, probs::Probabilities)
    q, base = e.q, e.base
    q < 0 && throw(ArgumentError("Order of Renyi entropy must be ≥ 0."))
    non0_probs = Iterators.filter(!iszero, probs.p)
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

entropy_maximum(e::Renyi, L::Int) = log_with_base(e.base)(L)

"""
    Shannon(base = 2)

The Shannon[^Shannon1948] entropy, used with [`entropy`](@ref) to compute:

```math
H(p) = - \\sum_i p[i] \\log(p[i])
```
with the ``\\log`` at the given `base`.

`Shannon(base)` is syntactically equivalent to `Renyi(; base)`.

[^Shannon1948]: C. E. Shannon, Bell Systems Technical Journal **27**, pp 379 (1948)
"""
Shannon(base = 2) = Renyi(; base)
