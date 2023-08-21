export ShannonExtropy

"""
    ShannonExtropy <: InformationMeasure
    ShannonExtropy(; base = 2)

The Shannon extropy (Lad et al., 2015[^Lad2015]), used with [`extropy`](@ref) to compute

```math
J(x) -\\sum_{i=1}^N (1 - p[i]) \\log{(1 - p[i])},
```

for a probability distribution ``P = \\{p_1, p_2, \\ldots, p_N\\}``,
with the ``\\log`` at the given `base`.

[^Lad2015]:
    Lad, F., Sanfilippo, G., & Agro, G. (2015). Extropy: Complementary dual of entropy.
"""
Base.@kwdef struct ShannonExtropy{B} <: InformationMeasure
    base::B = 2
end

function information(e::ShannonExtropy, probs::Probabilities)
    non0_probs = collect(Iterators.filter(!iszero, vec(probs)))
    if length(non0_probs) == 1
        return 0.0
    end

    return -sum((1 - pᵢ) * log(e.base, 1 - pᵢ) for pᵢ in non0_probs)
end

function information_maximum(e::ShannonExtropy, L::Int)
    if L == 1
        return 0.0
    end

    return (L - 1) * log(e.base, L / (L - 1))
end
