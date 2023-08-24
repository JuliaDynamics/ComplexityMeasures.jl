export RenyiExtropy

"""
    RenyiExtropy <: InformationMeasure
    RenyiExtropy(; q = 1.0, base = 2)

The Rényi extropy [Liu2023](@cite).

## Description

`RenyiExtropy` is used with [`extropy`](@ref) to compute

```math
J_R(P) = \\dfrac{-(n - 1) \\log{(n - 1)} + (n - 1) \\log{ \\left( \\sum_{i=1}^N {(1 - p[i])}^q \\right)} }{q - 1}
```

for a probability distribution ``P = \\{p_1, p_2, \\ldots, p_N\\}``,
with the ``\\log`` at the given `base`. Alternatively, `RenyiExtropy` can be used
with [`extropy_normalized`](@ref), which ensures that the computed extropy is
on the interval ``[0, 1]`` by normalizing to to the maximal Rényi extropy, given by

```math
J_R(P) = (N - 1)\\log \\left( \\dfrac{n}{n-1} \\right) .
```
"""
struct RenyiExtropy{Q,B} <: InformationMeasure
    q::Q
    base::B
end
RenyiExtropy(q; base = 2) = RenyiExtropy(q, base)
RenyiExtropy(; q = 1.0, base = 2) = RenyiExtropy(q, base)

function information(e::RenyiExtropy, probs::Probabilities)
    (; q, base) = e
    non0_probs = collect(Iterators.filter(!iszero, vec(probs)))

    if length(non0_probs) == 1
        return 0.0
    end

    if q ≈ 1
        return information(ShannonExtropy(; base), Probabilities(non0_probs))
    else
        N = length(non0_probs)
        num = -(N - 1)*log(base, N - 1) + (N - 1)*log(base, sum((1 - pᵢ)^q for pᵢ in non0_probs))
        den = 1 - q
        return num / den
    end
end

function information_maximum(e::RenyiExtropy, L::Int)
    (; q, base) = e

    if L == 1
        return 0.0
    end

    return (L - 1) * log(base, L / (L - 1))
end
