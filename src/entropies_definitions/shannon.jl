export Shannon

"""
    Shannon <: EntropyDefinition
    Shannon(; base = 2)

The Shannon[^Shannon1948] entropy, used with [`entropy`](@ref) to compute:

```math
H(p) = - \\sum_i p[i] \\log(p[i])
```
with the ``\\log`` at the given `base`.

[^Shannon1948]: C. E. Shannon, Bell Systems Technical Journal **27**, pp 379 (1948)
"""
Base.@kwdef struct Shannon{B} <: EntropyDefinition
    base::B = 2
end

function entropy(e::Shannon, probs::Probabilities)
    base = e.base
    non0_probs = Iterators.filter(!iszero, probs.p)
    logf = log_with_base(base)
    return -sum(x*logf(x) for x in non0_probs)
end

entropy_maximum(e::Shannon, L::Int) = log_with_base(e.base)(L)
