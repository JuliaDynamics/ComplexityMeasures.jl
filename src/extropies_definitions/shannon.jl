export ShannonExtropy

"""
    ShannonExtropy <: ExtropyDefinition
    ShannonExtropy(; base = 2)

The Shannon extropy (Lad et al., 2015[^Lad2015]), used with [`extropy`](@ref) to compute

```math
J(x) -\\sum_{i=1}^N (1 - p[i]) \\log{(1 - p[i])},
```

with the ``\\log`` at the given `base`.

[^Lad2015]:
    Lad, F., Sanfilippo, G., & Agro, G. (2015). Extropy: Complementary dual of entropy.
"""
Base.@kwdef struct ShannonExtropy{B} <: ExtropyDefinition
    base::B = 2
end

function extropy(e::ShannonExtropy, p::Probabilities)
    -sum((1 - pᵢ) * log(e.base, 1 - pᵢ) for pᵢ in p)
end

function extropy_maximum(e::ShannonExtropy, L::Int)
    return (L - 1) * log(e.base, L / (L - 1))
end
