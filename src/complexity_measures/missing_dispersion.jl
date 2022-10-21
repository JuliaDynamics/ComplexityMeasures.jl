using DelayEmbeddings
export MissingDispersionPatterns

"""
    MissingDispersionPatterns <: ComplexityMeasure
    MissingDispersionPatterns(; est = Dispersion())

An estimator for the number of missing dispersion patterns (``N_{MDP}``), a complexity
measure which can be used to detect nonlinearity in time series (Zhou et al.,
2022)[^Zhou2022], using the dispersion pattern estimator `c`.

Used with [`complexity`](@ref) or [`complexity_normalized](@ref)`.

## Description

``N_{MDP}`` is computed by first symbolising each `xᵢ ∈ x`, then embedding the resulting
symbol sequence (using `est`), and computing the quantity

```math
N_{MDP} = L - N_{ODP},
```

where `L = alphabet_length(est)` (i.e. the total number of possible dispersion patterns),
and ``N_{ODP}`` is defined as the number of *occurring* dispersion patterns.

The normalized complexity measure ``N_{MDP}^N = (L - N_{ODP})/L`` is
computed. The authors recommend that
`alphabet_length(est.symbolization)^est.m << length(x) - est.m*est.τ + 1`` to avoid
undersampling.

## Usage

In Zhou et al. (2022), [`MissingDispersionPatterns`](@ref) is used to detect nonlinearity
in time series by comparing the ``N_{MDP}`` for a time series `x` to ``N_{MDP}`` values for
an ensemble of surrogates of `x`. If ``N_{MDP} > q_{MDP}^{WIAAFT}``, where
``q_{MDP}^{WIAAFT}`` is some `q`-th quantile of the surrogate ensemble, then it is
taken as evidence for nonlinearity.

See also: [`Dispersion`](@ref), [`ReverseDispersion`](@ref), [`alphabet_length`](@ref).

[^Zhou2022]: Zhou, Q., Shang, P., & Zhang, B. (2022). Using missing dispersion patterns
    to detect determinism and nonlinearity in time series data. Nonlinear Dynamics, 1-20.
"""
struct MissingDispersionPatterns{D} <: ComplexityMeasure
    est::D

    function MissingDispersionPatterns(est::D) where {D <: Union{Dispersion, ReverseDispersion}}
        new{D}(est)
    end
end

function count_nonoccurring(x::AbstractVector{T}, est) where T
    τ, m = est.τ, est.m

    # Symbolize and embed
    symbols = symbolize_for_dispersion(x, est)
    embedding = genembed(symbols, 0:τ:(m - 1)*τ)
    n_occuring = length(Set(embedding.data))

    L = alphabet_length(est)
    n_not_occurring = L - n_occuring

    return L, n_not_occurring
end

function complexity(c::MissingDispersionPatterns, x::AbstractVector{T}) where T
    return count_nonoccurring(x, c.est)[2]
end

function complexity_normalized(c::MissingDispersionPatterns, x::AbstractVector{T}) where T
    L, n_not_occurring = count_nonoccurring(x, c.est)
    return n_not_occurring / L
end
