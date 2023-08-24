using DelayEmbeddings
using ComplexityMeasures: Dispersion
using ComplexityMeasures: total_outcomes, missing_outcomes

export MissingDispersionPatterns

"""
    MissingDispersionPatterns <: ComplexityEstimator
    MissingDispersionPatterns(est = Dispersion())

An estimator for the number of missing dispersion patterns (``N_{MDP}``), a complexity
measure which can be used to detect nonlinearity in time series [Zhou2023](@cite).

Used with [`complexity`](@ref) or [`complexity_normalized`](@ref), whose implementation
uses [`missing_outcomes`](@ref).

## Description

If used with [`complexity`](@ref), ``N_{MDP}`` is computed by first symbolising each
`xᵢ ∈ x`, then embedding the resulting symbol sequence using the dispersion pattern
estimator `est`, and computing the quantity

```math
N_{MDP} = L - N_{ODP},
```

where `L = total_outcomes(est)` (i.e. the total number of possible dispersion patterns),
and ``N_{ODP}`` is defined as the number of *occurring* dispersion patterns.

If used with [`complexity_normalized`](@ref), then ``N_{MDP}^N = (L - N_{ODP})/L`` is
computed. The authors recommend that
`total_outcomes(est.symbolization)^est.m << length(x) - est.m*est.τ + 1` to avoid
undersampling.

!!! note "Encoding"
    [`Dispersion`](@ref)'s linear mapping from CDFs to integers is based on equidistant
    partitioning of the interval `[0, 1]`. This is slightly different from Zhou et
    al. (2023)[Zhou2023](@ref), which uses the linear mapping ``s_i := \\text{round}(y + 0.5)``.

## Usage

In Zhou et al. (2023)[Zhou2023](@ref), [`MissingDispersionPatterns`](@ref) is used to detect nonlinearity
in time series by comparing the ``N_{MDP}`` for a time series `x` to ``N_{MDP}`` values for
an ensemble of surrogates of `x`. If ``N_{MDP} > q_{MDP}^{WIAAFT}``, where
``q_{MDP}^{WIAAFT}`` is some `q`-th quantile of the surrogate ensemble, then it is
taken as evidence for nonlinearity.

See also: [`Dispersion`](@ref), [`ReverseDispersion`](@ref), [`total_outcomes`](@ref).
"""
Base.@kwdef struct MissingDispersionPatterns{D} <: ComplexityEstimator
    est::D = Dispersion()
end

function complexity(c::MissingDispersionPatterns, x::AbstractVector{T}) where T
    return missing_outcomes(c.est, x)
end

function complexity_normalized(c::MissingDispersionPatterns, x::AbstractVector{T}) where T
    return missing_outcomes(c.est, x) / total_outcomes(c.est, x)
end
