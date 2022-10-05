using DelayEmbeddings
export missing_dispersion

"""
    missing_dispersion(x::AbstractVector{T}; est::Dispersion = Dispersion(),
        normalize = true) → NMDP::Real

Calculate the (normalized, if `normalized == true`) number of missing dispersion
patterns (``N_{MDP}``) for `x`, a complexity measure which can be used to detect
nonlinearity in time series (Zhou et al., 2022)[^Zhou2022].

## Description

``N_{MDP}`` is computed by first symbolising each `xᵢ ∈ x`, then embedding the resulting
symbol sequence (using `est`), and computing the quantity

```math
N_{MDP} = L - N_{ODP},
```

where `L = alphabet_length(est)` (i.e. the total number of possible dispersion patterns),
and ``N_{ODP}`` is defined as the number of *occurring* dispersion patterns.

If `normalize == true`, then the quantity ``N_{MDP}^N = \\dfrac{L - N_{ODP}}{L}`` is
computed. The authors recommend that
`est.symbolization.c^est.m << length(x) - est.m*est.τ + 1`` to avoid undersampling.

See also: [`Dispersion`](@ref), [`alphabet_length`](@ref), [`reverse_dispersion`](@ref).

[^Zhou2022]: Zhou, Q., Shang, P., & Zhang, B. (2022). Using missing dispersion patterns
    to detect determinism and nonlinearity in time series data. Nonlinear Dynamics, 1-20.
"""
function missing_dispersion(x::AbstractVector{T}; est::Dispersion = Dispersion(),
        normalize::Bool = false) where T
    τ, m = est.τ, est.m

    # Symbolize and embed
    symbols = _symbolize_for_dispersion(x, est)
    embedding = genembed(symbols, 0:τ:(m - 1)*τ)
    n_occuring = length(Set(embedding.data))

    L = alphabet_length(est)
    n_not_occurring = L - n_occuring

    if (normalize)
        return n_not_occurring / L
    else
        # Converting to float ensures return type stability.
        return float(n_not_occurring)
    end
end
