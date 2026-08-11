export Tsallis

"""
    Tsallis <: InformationMeasure
    Tsallis(q; k = 1.0, base = 2)
    Tsallis(; q = 1.0, k = 1.0, base = 2)

The Tsallis generalized order-`q` entropy [Tsallis1988](@cite), used with
[`information`](@ref) to compute an entropy.

## Description

The Tsallis entropy is a generalization of the Boltzmann-Gibbs entropy, with `k` standing
for the Boltzmann constant. It is the probability-weighted mean of the ``q``-deformed
logarithm,

```math
S_q(p) = k \\sum_i p[i] \\log_q{\\left( 1/p[i] \\right)},
```
```math
\\log_q{(x)} = \\dfrac{x^{1 - q} - 1}{(1 - q)\\ln{(base)}},
```

with the ``q``-logarithm at the given `base`. Equivalently, in closed form,
``S_q(p) = k \\left(1 - \\sum_i p[i]^q\\right) / ((q - 1)\\ln{(base)})``. As ``q \\to 1``,
``\\log_q`` becomes the ordinary logarithm to the given `base`, so `q == 1` is exactly
the [`Shannon`](@ref) entropy to that same base.

`base` sets the (dimensionless) scale of the returned value: the expression published by
[Tsallis1988](@citet) divided by ``\\ln{(base)}``. Use `base = MathConstants.e` to recover
that expression exactly. See [Units and the `base` keyword](@ref units_and_base).

The [`self_information`](@ref) is ``I_q(p_i) = k \\log_q{(1/p_i)}``.

The maximum value of the Tsallis entropy is
``k(L^{1 - q} - 1)/((1 - q)\\ln{(base)})``, with ``L`` the [`total_outcomes`](@ref).
"""
struct Tsallis{Q,K,B} <: Entropy
    q::Q
    k::K
    base::B
end
Tsallis(q; k=1.0, base=2) = Tsallis(q, k, base)
Tsallis(; q=1.0, k=1.0, base=2) = Tsallis(q, k, base)

function information(e::Tsallis, probs::Probabilities)
    (; q, k, base) = e
    # As for Renyi, we want to skip the zeros as well.
    non0_probs = Iterators.filter(!iszero, probs.p)
    if q ≈ 1
        lb = log_with_base(base)
        return -k * sum(p * lb(p) for p in non0_probs)
    else
        # The q-logarithm is taken to the given `base`, i.e. scaled by 1/log(base), so
        # that the q → 1 limit is continuous with the Shannon branch above.
        return k / ((q - 1) * log(base)) * (1 - sum(p^q for p in non0_probs))
    end
end

function information_maximum(e::Tsallis, L::Int)
    (; q, k, base) = e
    if q ≈ 1.0
        return k * log_with_base(base)(L)
    else
        return k * (L^(1 - q) - 1) / ((1 - q) * log(base))
    end
end

function self_information(e::Tsallis, pᵢ, N=nothing)
    (; q, k, base) = e
    # Mirror the `q ≈ 1` branch of `information`, which reduces to Shannon.
    q ≈ 1 && return -k * log_with_base(base)(pᵢ)
    return k * (1 - pᵢ^(q - 1)) / ((q - 1) * log(base))
end
