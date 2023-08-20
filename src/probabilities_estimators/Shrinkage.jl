export Shrinkage

# TODO: make sure we act correctly for `probabilities` and `allprobabilities`.
"""
    Shrinkage{<:OutcomeSpace} <: ProbabilitiesEstimator
    Shrinkage(model::OutcomeSpace; t = nothing, λ = nothing)

The `Shrinkage` estimator is used with [`probabilities`](@ref) and related functions
to estimate probabilities over the given `m`-element counting-based
[`OutcomeSpace`](@ref) using James-Stein-type shrinkage
(James & Stein, 1961)[^JamesStein1961], as presented in
Hausser & Strimmer (2009)[^Hausser2009].

## Description

The `Shrinkage` estimator estimates a cell probability ``\\theta_{k}^{\\text{Shrink}}`` as

```math
\\theta_{k}^{\\text{Shrink}} = \\lambda t_k + (1-\\lambda) \\hat{\\theta}_k^{RelativeAmount},
```

where ``\\lambda \\in [0, 1]`` is the shrinkage intensity (``\\lambda = 0`` means
no shrinkage, and ``\\lambda = 1`` means full shrinkage), and ``t_k`` is the shrinkage
target. Hausser & Strimmer (2009)[^Hausser2009] picks ``t_k = 1/m``, i.e. the uniform
distribution.

If `t == nothing`, then ``t_k`` is set to ``1/m`` for all ``k``,
as in Hausser & Strimmer (2009)[^Hausser2009].
If `λ == nothing` (the default), then the shrinkage intensity is optimized according
to Hausser & Strimmer (2009)[^Hausser2009]. Hence, you should probably not pick
`λ` nor `t` manually, unless you know what you are doing.

## Assumptions

The `Shrinkage` estimator assumes a fixed and known number of outcomes `m`. Thus, using
it with [`probabilities`](@ref) and [`allprobabilities`](@ref) will yield different results,
depending on whether all outcomes are observed in the input data or not.
For [`probabilities`](@ref), `m` is the number of *observed* outcomes.
For [`allprobabilities`](@ref), `m = total_outcomes(o, x)`, where `o` is the
[`OutcomeSpace`](@ref) and `x` is the input data.

!!! note
    If used with [`allprobabilities`](@ref)/[`allprobabilities_and_outcomes`](@ref), then
    outcomes which have not been observed may be assigned non-zero probabilities.
    This might affect your results if using e.g. [`missing_outcomes`](@ref).

## Examples

```julia
using ComplexityMeasures
x = cumsum(randn(100))
ps_shrink = probabilities(Shrinkage(OrdinalPatterns(m = 3)), x)
```

See also: [`RelativeAmount`](@ref), [`BayesianRegularization`](@ref).

[^JamesStein1961]:
    James, W., & Stein, C. (1992). Estimation with quadratic loss. In Breakthroughs in
    statistics: Foundations and basic theory (pp. 443-460). New York, NY: Springer New York.
[^Hausser2009]:
    Hausser, J., & Strimmer, K. (2009). Entropy inference and the James-Stein estimator,
    with application to nonlinear gene association networks. Journal of Machine Learning
    Research, 10(7).
"""
struct Shrinkage{O <: OutcomeSpace, T <: Union{Nothing, Real, Vector{<:Real}}, L <: Union{Nothing, Real}} <: ProbabilitiesEstimator
    outcomemodel::O
    t::T
    λ::L
    function Shrinkage(o::O, t::T, λ::L) where {O <: OutcomeSpace, T, L}
        verify_counting_based(o, "Shrinkage")
        new{O, T, L}(o, t, λ)
    end
end

Shrinkage(o::OutcomeSpace; t = nothing, λ = nothing) = Shrinkage(o, t, λ)

function probabilities_and_outcomes(est::Shrinkage, x)
    probs, Ω = probabilities_and_outcomes(est.outcomemodel, x)
    return probs_and_outs_from_histogram(est, probs, Ω, x)
end

# TODO: this doesn't work for SymbolicDispersion estimator.
function allprobabilities_and_outcomes(est::Shrinkage, x)
    probs_all, Ω_all = allprobabilities_and_outcomes(est.outcomemodel, x)
    return probs_and_outs_from_histogram(est, probs_all, Ω_all, x)
end

function probs_and_outs_from_histogram(est::Shrinkage, probs_observed, Ω_observed, x)
    (; outcomemodel, t) = est

    n = encoded_space_cardinality(outcomemodel, x) # Normalize based on *encoded* data.
    m = length(Ω_observed)
    Ω = outcomes(outcomemodel, x)
    if t isa Vector{<:Real}
        length(t) == M || throw(DimensionMismatch("If `t` is a vector, `length(t)` must equal the number of elements in the outcome space (got $M outcomes, but length(t)=$(length(t)))."))
    end
    λ = get_λ(est, n, probs_observed, t, m)
    @assert 0 ≤ λ ≤ 1

    probs = zeros(m)
    for (k, ωₖ) in enumerate(Ω_observed)
        tₖ = get_tₖ(t, k, m)
        pₖ = θₖ_shrink(probs_observed[k], λ, tₖ)
        idx = findfirst(x -> x == Ω_observed[k], Ω_observed)
        probs[idx] = θₖ_shrink(probs_observed[k], λ, tₖ)
    end
    @assert sum(probs) ≈ 1.0
    return Probabilities(probs), Ω_observed
end

function get_λ(est, n, probs_observed, t, m)
    # Optimal shrinkage intensity (eq. 5 in Hausser and Strimmer, 2009).
    if est.λ === nothing
        densum = 0.0
        for k = 1:m
            tₖ = get_tₖ(t, k, m)
            densum += (tₖ - probs_observed[k]) ^ 2
        end
        λ = (1 - sum(probs_observed .^ 2)) / (n - 1)*densum
    # User-picked shrinkage intensity.
    else
        λ = est.λ
    end
    # Truncate, so that 0 ≤ λ ≤ 1.
    return max(0.0, min(1.0, λ))
end

function get_tₖ(t, k::Int, m::Int)
    if t isa Real
        return t
    elseif t === nothing
        return 1/m
    else
        return t[k]
    end
end
θₖ_shrink(θₖML, λ, tₖ) = λ*tₖ + (1 - λ)*θₖML
