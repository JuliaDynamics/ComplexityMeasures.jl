using DelayEmbeddings
export MissingDispersionPatterns

"""
    MissingDispersionPatterns <: ComplexityMeasure
    MissingDispersionPatterns(est = Dispersion())

An estimator for the number of missing dispersion patterns (``N_{MDP}``), a complexity
measure which can be used to detect nonlinearity in time series (Zhou et al.,
2022)[^Zhou2022].

Used with [`complexity`](@ref) or [`complexity_normalized`](@ref).

## Description

If used with [`complexity`](@ref), ``N_{MDP}`` is computed by first symbolising each
`xᵢ ∈ x`, then embedding the resulting symbol sequence using the dispersion pattern
estimator `est`, and computing the quantity

```math
N_{MDP} = L - N_{ODP},
```

where `L = alphabet_length(est)` (i.e. the total number of possible dispersion patterns),
and ``N_{ODP}`` is defined as the number of *occurring* dispersion patterns.

If used with [`complexity_normalized`](@ref), then ``N_{MDP}^N = (L - N_{ODP})/L`` is
computed. The authors recommend that
`alphabet_length(est.symbolization)^est.m << length(x) - est.m*est.τ + 1` to avoid
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
Base.@kwdef struct MissingDispersionPatterns{D} <: ComplexityMeasure
    est::D = Dispersion()
end

function count_nonoccurring(x::AbstractVector{T}, est) where T
    τ, m = est.τ, est.m
    probs = probabilities(x, est)
    L = alphabet_length(x, est)
    O = count(!iszero, probs)
    return L - O
end

function complexity(c::MissingDispersionPatterns, x::AbstractVector{T}) where T
    return count_nonoccurring(x, c.est)
end

function complexity_normalized(c::MissingDispersionPatterns, x::AbstractVector{T}) where T
    NO = count_nonoccurring(x, c.est)
    return NO / alphabet_length(x, c.est)
end
