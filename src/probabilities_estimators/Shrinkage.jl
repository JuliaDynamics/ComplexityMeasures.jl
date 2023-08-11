export Shrinkage

"""
    Shrinkage{<:OutcomeSpace} <: ProbabilitiesEstimator
    Shrinkage(model::OutcomeSpace, t = nothing, λ = nothing)

The `Shrinkage` estimator is used with [`probabilities`](@ref) and related functions
to estimate probabilities over the given `m`-element [`OutcomeSpace`](@ref) using
James-Stein-type shrinkage (James & Stein, 1961)[^JamesStein1961], as presented in
Hausser & Strimmer (2009)[^Hausser2009]. See [`ProbabilitiesEstimator`](@ref) for usage.

## Description

The `Shrinkage` estimator estimates a cell probability ``\\theta_{k}^{\\text{Shrink}}`` as

```math
\\theta_{k}^{\\text{Shrink}} = \\lambda t_k + (1-\\lambda) \\hat{\\theta}_k^{MLE},
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

## Examples

```julia
using ComplexityMeasures
x = cumsum(randn(100))
ps_shrink = probabilities(Shrinkage(SymbolicPermutation(m = 3)), x)
```

See also: [`MLE`](@ref), [`Bayes`](@ref).

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
end

Shrinkage(o::OutcomeSpace; t = nothing, λ = nothing) = Shrinkage(o, t, λ)

# We need to implement `probabilities_and_outcomes` and `allprobabilities` separately,
# because the number of elements in the outcome space determines the factor `A`, since
# A = sum(aₖ). Explicitly modelling the entire outcome space, instead of considering
# only the observed outcomes, will therefore affect the estimated probabilities.
function probabilities_and_outcomes(est::Shrinkage, x)
    t = est.t
    M = total_outcomes(est.outcomemodel, x)
    # Normalization factor is based on the *encoded* data.
    n = encoded_space_cardinality(est.outcomemodel, x)

    # Maximum likelihood estimates for the *observed* outcomes. Unobserved outcomes
    # are just ignored, so we need to manually add them
    observed_freqs, observed_Ω = counts_and_outcomes(est.outcomemodel, x)
    M_observed = length(observed_Ω)
    Ω = outcomes(est.outcomemodel, x)

    θMLs = zeros(M) # θMLs[i] is the maximum-likelihood estimate for the outcome Ω[i]
    for (i, Ωᵢ) in enumerate(Ω)
        if Ωᵢ ∉ observed_Ω
            θMLs[i] = 0.0
        else
            idx = findfirst(x -> x == Ωᵢ, observed_Ω)
            θMLs[i] = observed_freqs[idx] / sum(observed_freqs)
        end
    end

    λ = get_λ(est, n, θMLs, t, M)

    @assert 0 ≤ λ ≤ 1
    probs = zeros(M)
    for i in eachindex(Ω)
        tᵢ = get_tₖ(est.t, i, M_observed)
        probs[i] = θₖ_shrink(θMLs[i], λ, tᵢ)
    end
    @assert sum(probs) ≈ 1.0
    idxs = findall(x -> x > 0, probs)
    return Probabilities(probs[idxs]), observed_Ω[idxs]
end

function allprobabilities_and_outcomes(est::Shrinkage, x)
    t = est.t

    # Maximum likelihood estimates for *all* outcomes.
    θₖMLs, ΩMLs = allprobabilities_and_outcomes(est.outcomemodel, x)
    M = length(ΩMLs)

    if t isa Vector{<:Real}
        length(t) == M || throw(DimensionMismatch("If `t` is a vector, `length(t)` must equal the number of elements in the outcome space (got $M outcomes, but length(t)=$(length(t)))."))
    end

    # Normalization factor is based on the *encoded* data.
    n = encoded_space_cardinality(est.outcomemodel, x)
    λ = get_λ(est, n, θₖMLs, t, M)
    @assert 0 ≤ λ ≤ 1

    # Adjust the maximum likelihood estimates
    probs = zeros(M)
    for i in 1:M
        tᵢ = get_tₖ(t, i, M)
        Ωᵢ = ΩMLs[i]
        θᵢML = θₖMLs[i]
        idx = findfirst(x -> x == Ωᵢ, ΩMLs)
        probs[idx] = θₖ_shrink(θᵢML, λ, tᵢ)
    end

    @assert sum(probs) ≈ 1.0
    return Probabilities(probs), ΩMLs
end


function get_λ(est, n, θₖMLs, t, M)
    # Optimal shrinkage intensity (eq. 5 in Hausser and Strimmer, 2009).
    if est.λ === nothing
        densum = 0.0
        for k = 1:M
            tₖ = get_tₖ(t, k, M)
            densum += (tₖ - θₖMLs[k]) ^ 2
        end
        λ = (1 - sum(θₖMLs .^ 2)) / (n - 1)*densum
    # User-picked shrinkage intensity.
    else
        λ = est.λ
    end
    # Truncate, so that 0 ≤ λ ≤ 1.
    return max(0.0, min(1.0, λ))
end

function get_tₖ(t, k::Int, M::Int)
    if t isa Real
        return t
    elseif t === nothing
        return 1/M
    else
        return t[k]
    end
end
θₖ_shrink(θₖML, λ, tₖ) = λ*tₖ + (1 - λ)*θₖML
