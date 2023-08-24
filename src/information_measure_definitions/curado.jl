export Curado

"""
    Curado <: InformationMeasure
    Curado(; b = 1.0)

The Curado entropy (Curado & Nobre, 2004)[Curado2004](@cite), used with [`information`](@ref) to
compute

```math
H_C(p) = \\left( \\sum_{i=1}^N e^{-b p_i} \\right) + e^{-b} - 1,
```

with `b ∈ ℛ, b > 0`, and the terms outside the sum ensures that ``H_C(0) = H_C(1) = 0``.

The maximum entropy for Curado is ``L(1 - \\exp(-b/L)) + \\exp(-b) - 1`` with ``L``
the [`total_outcomes`](@ref).
"""
Base.@kwdef struct Curado{B} <: Entropy
    b::B = 1.0

    function Curado(b::B) where B <: Real
        b > 0 || throw(ArgumentError("Need b > 0. Got b=$(b)."))
        return new{B}(b)
    end
end

function information(e::Curado, probs::Probabilities)
    b = e.b
    return sum(1 - exp(-b*pᵢ)  for pᵢ in probs) + exp(-b) - 1
end

function information_maximum(e::Curado, L::Int)
    b = e.b
    # Maximized for the uniform distribution, which for distribution of length L is
    return L * (1 - exp(-b/L)) + exp(-b) - 1
end
