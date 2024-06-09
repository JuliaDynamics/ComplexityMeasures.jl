export Shannon

"""
    Shannon <: InformationMeasure
    Shannon(; base = 2)

The Shannon [Shannon1948](@cite) entropy, used with [`information`](@ref) to compute:

```math
H(p) = - \\sum_i p[i] \\log(p[i])
```
with the ``\\log`` at the given `base`.

The maximum value of the Shannon entropy is ``\\log_{base}(L)``, which is the entropy of the
uniform distribution with ``L`` the [`total_outcomes`](@ref).
"""
Base.@kwdef struct Shannon{B} <: Entropy
    base::B = 2
end

function information(e::Shannon, probs::Probabilities)
    non0_probs = Iterators.filter(!iszero, vec(probs))
    logf = log_with_base(e.base)
    return -sum(x*logf(x) for x in non0_probs)
end

function selfinformation(e::Shannon, pᵢ)
    return -log(e.base, pᵢ)
end

information_maximum(e::Shannon, L::Int) = log_with_base(e.base)(L)
