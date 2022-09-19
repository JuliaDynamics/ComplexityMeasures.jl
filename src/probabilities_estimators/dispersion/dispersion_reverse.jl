export ReverseDispersion
export entropy_reverse_dispersion
export distance_to_whitenoise

"""
    ReverseDispersion(; s = GaussianSymbolization(5), m = 2, τ = 1, check_unique = true,
        normalize = true)

A probability estimator that in Rostaghi & Azami, 2016[^Rostaghi2016] is used to compute the
"reverse dispersion entropy", which measures how far from being white noise a signal is.

For computing probabilities, [`ReverseDispersion`](@ref) is equivalent to
[`Dispersion`](@ref), and parameters have the same meaning. For computing entropies,
however, [`ReverseDispersion`](@ref) adds an additional step consisting of computation and
normalization of distances from white noise (see Li et al. 2019 for details).

See also: [`Dispersion`](@ref).

[^Li2019]: Li, Y., Gao, X., & Wang, L. (2019). Reverse dispersion entropy: a new
    complexity measure for sensor signal. Sensors, 19(23), 5203.
"""
Base.@kwdef struct ReverseDispersion <: ProbabilitiesEstimator
    s = GaussianSymbolization(n_categories = 5)
    m = 2
    τ = 1
    check_unique = false
    normalize = true
end

function probabilities(x::AbstractVector, est::ReverseDispersion)
    probabilities(x, Dispersion(m = est.m, τ = est.τ, s = est.s,
        check_unique = est.check_unique, normalize = est.normalize))
end


function distance_to_whitenoise(𝐩::Probabilities, N, m)
    # We can safely skip non-occurring symbols, because they don't contribute
    # to the sum in eq. 3 in Li et al. (2019)
    return sum(𝐩[i]^2 for i in eachindex(𝐩)) - 1/(N^m)
end

function renyi_entropy(x::AbstractVector{T}, est::ReverseDispersion;
        q = 1, base = 2) where T <: Real

    𝐩 = probabilities(x, est)
    Hrde = distance_to_whitenoise(𝐩, s.n_categories, est.m)

    if normalize
        # The factor `f` considers *all* possible symbols (also non-occurring)
        f = s.n_categories^m
        return Hrde / (1 - (1/f))
    else
        return Hrde
    end
end
