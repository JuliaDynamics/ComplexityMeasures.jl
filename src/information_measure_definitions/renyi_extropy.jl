export RenyiExtropy

"""
    RenyiExtropy <: ProbabilitiesFunctional
    RenyiExtropy(; q = 1.0, base = 2)

The Rényi extropy (Liu & Xiao, 2021[^Liu2021]).

## Description

`RenyiExtropy` is used [`extropy`](@ref) to compute

```math
J_R(P) = \\dfrac{-(n - 1)\\log(n-1) + (n-1) \\log\\( \\sum_{i=1}^N (1 - p[i])^q \\)}{q - 1}
```

for a probability distribution ``P = \\{p_1, p_2, \\ldots, p_N\\}``,
with the ``\\log`` at the given `base`. Alternatively, `RenyiExtropy` can be used
with [`extropy_normalized`](@ref), which ensures that the computed extropy is
on the interval ``[0, 1]`` by normalizing to to the maximal Rényi extropy, given by

```math
J_R(P) = (N - 1)\\log \\( \\dfrac{n}{n-1} \\).
```

[^Liu2021]:
    Liu, J., & Xiao, F. (2021). Renyi extropy. Communications in Statistics-Theory and
    Methods, 1-12.
"""
struct RenyiExtropy{Q,B} <: InformationMeasureDefinition
    q::Q
    base::B
end
RenyiExtropy(q; base = 2) = RenyiExtropy(q, base)
RenyiExtropy(; q = 1.0, base = 2) = RenyiExtropy(q, base)

function information(e::RenyiExtropy, probs::Probabilities)
    (; q, base) = e

    if length(probs) == 1
        return 0.0
    end

    if q ≈ 1
        return information(ShannonExtropy(; base), probs)
    else
        N = length(probs)
        num = -(N - 1)*log(base, N - 1) + (N - 1)*log(base, sum((1 - pᵢ)^q for pᵢ in probs))
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
