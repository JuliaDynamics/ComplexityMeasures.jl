using DelayEmbeddings
using ComplexityMeasures: Dispersion
using ComplexityMeasures: total_outcomes, missing_outcomes

export MissingDispersionPatterns

"""
    MissingDispersionPatterns <: ComplexityEstimator
    MissingDispersionPatterns(est = Dispersion()) → mdp

An estimator for the number of missing dispersion patterns (``N_{MDP}``), a complexity
measure which can be used to detect nonlinearity in time series [Zhou2023](@cite).

Used with [`complexity`](@ref) or [`complexity_normalized`](@ref), whose implementation
uses [`missing_outcomes`](@ref).

## Description

When used with [`complexity`](@ref), `complexity(mdp)` is syntactically equivalent
with just [`missing_outcomes`](@ref)`(est)`. When used with [`complexity_normalized`](@ref),
we further divide `missing_outcomes(est)/total_outcomes(est)`.

!!! note "Encoding"
    [`Dispersion`](@ref)'s linear mapping from CDFs to integers is based on equidistant
    partitioning of the interval `[0, 1]`. This is slightly different from
    [Zhou2023](@citet), which uses the linear mapping ``s_i := \\text{round}(y + 0.5)``.

## Usage

In [Zhou2023](@citet), [`MissingDispersionPatterns`](@ref) is used to detect nonlinearity
in time series by comparing the MDP for a time series `x` to values for
an ensemble of surrogates of `x`, as per the standard analysis of TimeseriesSurrogates.jl.
If the MDP value of ``x`` is significantly larger than some high quantile of the surrogate
distribution, then it is taken as evidence for nonlinearity.

See also: [`Dispersion`](@ref), [`ReverseDispersion`](@ref), [`total_outcomes`](@ref).
"""
Base.@kwdef struct MissingDispersionPatterns{D} <: ComplexityEstimator
    est::D = Dispersion()
end

function complexity(c::MissingDispersionPatterns, x)
    return missing_outcomes(c.est, x)
end

function complexity_normalized(c::MissingDispersionPatterns, x)
    return missing_outcomes(c.est, x) / total_outcomes(c.est, x)
end
