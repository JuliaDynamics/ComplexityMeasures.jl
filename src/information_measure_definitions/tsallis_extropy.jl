export TsallisExtropy

"""
    TsallisExtropy <: InformationMeasure
    TsallisExtropy(q; k = 1.0, base = 2)
    TsallisExtropy(; q = 1.0, k = 1.0, base = 2)

The Tsallis extropy [Xue2023](@cite).

## Description

`TsallisExtropy` is used with [`information`](@ref) to compute

```math
J_T(P) = \\dfrac{k}{\\ln{(base)}} \\dfrac{N - 1 - \\sum_{i=1}^N ( 1 - p[i])^q}{q - 1}
```

for a probability distribution ``P = \\{p_1, p_2, \\ldots, p_N\\}``, with `q == 1` giving
exactly the [`ShannonExtropy`](@ref) at the same base.

`base` sets the (dimensionless) scale of the returned value: the expressions published by
[Xue2023](@citet) divided by ``\\ln{(base)}``. Use `base = MathConstants.e` to recover
them exactly. Since [`Tsallis`](@ref) is scaled the same way, the identity
``H_T = J_T`` for two-element distributions holds at every base.
See [Units and the `base` keyword](@ref units_and_base).

Alternatively, `TsallisExtropy` can be used
with [`information_normalized`](@ref), which ensures that the computed extropy is
on the interval ``[0, 1]`` by normalizing to to the maximal Tsallis extropy, given by

```math
J_T(P) = \\dfrac{k}{\\ln{(base)}} \\dfrac{(N - 1)N^{q - 1} - (N - 1)^q}{(q - 1)N^{q - 1}}
```
"""
struct TsallisExtropy{Q, K, B} <: InformationMeasure
    q::Q
    k::K
    base::B
end
TsallisExtropy(q; k = 1.0, base = 2) = TsallisExtropy(q, k, base)
TsallisExtropy(; q = 1.0, k = 1.0, base = 2) = TsallisExtropy(q, k, base)

function information(e::TsallisExtropy, probs::Probabilities)
    (; q, k, base) = e
    non0_probs = collect(Iterators.filter(!iszero, vec(probs)))

    if length(non0_probs) == 1
        return 0.0
    end

    if q ≈ 1
        return information(ShannonExtropy(; base), Probabilities(non0_probs))
    else
        N = length(non0_probs)
        # As for `Tsallis`, scaling by 1/log(base) makes the q → 1 limit continuous with
        # the `ShannonExtropy` branch above.
        c = k / ((q - 1) * log(base))
        return c * (N - 1 - sum((1 - pᵢ)^q for pᵢ in non0_probs))
    end
end

function information_maximum(e::TsallisExtropy, L::Int)
    (; q, k, base) = e

    if L == 1
        return 0.0
    end

    return k * ((L - 1) * L^(q - 1) - (L - 1)^q) / ((q - 1) * L^(q - 1) * log(base))
end

function self_information(e::TsallisExtropy, pᵢ, N) #must have N
    (; q, k, base) = e
    return k * ((N - 1) - (1 - pᵢ)^q) / ((q - 1) * log(base))
end