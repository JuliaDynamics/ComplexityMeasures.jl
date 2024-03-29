export Shrinkage

"""
    Shrinkage{<:OutcomeSpace} <: ProbabilitiesEstimator
    Shrinkage(; t = nothing, λ = nothing)

The `Shrinkage` estimator is used with [`probabilities`](@ref) and related functions
to estimate probabilities over the given `m`-element counting-based
[`OutcomeSpace`](@ref) using James-Stein-type shrinkage
[JamesStein1992](@cite), as presented in [Hausser2009](@citet).

## Description

The `Shrinkage` estimator estimates a cell probability ``\\theta_{k}^{\\text{Shrink}}`` as

```math
\\theta_{k}^{\\text{Shrink}} = \\lambda t_k + (1-\\lambda) \\hat{\\theta}_k^{RelativeAmount},
```

where ``\\lambda \\in [0, 1]`` is the shrinkage intensity (``\\lambda = 0`` means
no shrinkage, and ``\\lambda = 1`` means full shrinkage), and ``t_k`` is the shrinkage
target. [Hausser2009](@citet) picks ``t_k = 1/m``, i.e. the uniform
distribution.

If `t == nothing`, then ``t_k`` is set to ``1/m`` for all ``k``,
as in [Hausser2009](@citet).
If `λ == nothing` (the default), then the shrinkage intensity is optimized according
to [Hausser2009](@citet). Hence, you should probably not pick
`λ` nor `t` manually, unless you know what you are doing.

## Assumptions

The `Shrinkage` estimator assumes a fixed and known number of outcomes `m`. Thus, using
it with [`probabilities_and_outcomes`](@ref)) and 
[`allprobabilities_and_outcomes`](@ref) will yield different results,
depending on whether all outcomes are observed in the input data or not.
For [`probabilities_and_outcomes`](@ref), `m` is the number of *observed* outcomes.
For [`allprobabilities_and_outcomes`](@ref), `m = total_outcomes(o, x)`, where `o` is the
[`OutcomeSpace`](@ref) and `x` is the input data.

!!! note
    If used with [`allprobabilities_and_outcomes`](@ref), then
    outcomes which have not been observed may be assigned non-zero probabilities.
    This might affect your results if using e.g. [`missing_outcomes`](@ref).

## Examples

```julia
using ComplexityMeasures
x = cumsum(randn(100))
ps_shrink = probabilities(Shrinkage(), OrdinalPatterns{3}(), x)
```

See also: [`RelativeAmount`](@ref), [`BayesianRegularization`](@ref).
"""
struct Shrinkage{T <: Union{Nothing, Real, Vector{<:Real}}, L <: Union{Nothing, Real}} <: ProbabilitiesEstimator
    t::T
    λ::L
    function Shrinkage(; t::T = nothing, λ::L = nothing) where {T, L}
        new{T, L}(t, λ)
    end
end

function probabilities_and_outcomes(est::Shrinkage, outcomemodel::OutcomeSpace, x)
    probs, Ω = probabilities_and_outcomes(RelativeAmount(), outcomemodel, x)
    return probs_and_outs_from_histogram(est, outcomemodel, probs, Ω, x)
end

function allprobabilities_and_outcomes(est::Shrinkage, outcomemodel::OutcomeSpace, x)
    probs_all, Ω_all = allprobabilities_and_outcomes(outcomemodel, x)
    return probs_and_outs_from_histogram(est, outcomemodel, probs_all, Ω_all, x)
end

function probs_and_outs_from_histogram(est::Shrinkage, outcomemodel::OutcomeSpace,
        probs_observed, Ω_observed, x)
    verify_counting_based(outcomemodel, "Shrinkage")
    t = est.t

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
    p = Probabilities(probs, Ω_observed,)
    return p, outcomes(p)
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
