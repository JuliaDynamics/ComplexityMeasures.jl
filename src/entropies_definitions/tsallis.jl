export Tsallis

"""
    Tsallis <: EntropyDefinition
    Tsallis(q; k = 1.0, base = 2)
    Tsallis(; q = 1.0, k = 1.0, base = 2)

The Tsallis[^Tsallis1988] generalized order-`q` entropy, used with [`entropy`](@ref)
to compute an entropy.

`base` only applies in the limiting case `q == 1`, in which the Tsallis entropy reduces
to Shannon entropy.

## Description

The Tsallis entropy is a generalization of the Boltzmann-Gibbs entropy,
with `k` standing for the Boltzmann constant. It is defined as

```math
S_q(p) = \\frac{k}{q - 1}\\left(1 - \\sum_{i} p[i]^q\\right)
```

If the probability estimator has known alphabet length ``L``, then the maximum
value of the Tsallis entropy is ``k(L^{1 - q} - 1)/(1 - q)``.

[^Tsallis1988]:
    Tsallis, C. (1988). Possible generalization of Boltzmann-Gibbs statistics.
    Journal of statistical physics, 52(1), 479-487.
"""
struct Tsallis{Q,K,B} <: EntropyDefinition
    q::Q
    k::K
    base::B
end
Tsallis(q; k = 1.0, base = 2) = Tsallis(q, k, base)
Tsallis(; q = 1.0, k = 1.0, base = 2) = Tsallis(q, k, base)

function entropy(e::Tsallis, probs::Probabilities)
    (; q, k, base) = e
    # As for Renyi, we want to skip the zeros as well.
    non0_probs = Iterators.filter(!iszero, probs.p)
    if q ≈ 1
        return -sum(p * log(base, p) for p in non0_probs)
    else
        return k/(q-1)*(1 - sum(p^q for p in non0_probs))
    end
end

function entropy_maximum(e::Tsallis, L::Int)
    (; q, k, base) = e
    if q ≈ 1.0
        return log_with_base(base)(L)
    else
        return k*(L^(1 - q) - 1) / (1 - q)
    end
end
