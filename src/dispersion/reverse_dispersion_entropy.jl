export ReverseDispersion
export entropy_reverse_dispersion
export distance_to_whitenoise

"""
    ReverseDispersion(; s = GaussianSymbolization(5), m = 2, τ = 1, check_unique = true)

A probability estimator using the reverse dispersion entropy technique from
Li et al. (2019)[^Li2019].

Although the reverse dispersion entropy is not intended as a probability estimator per se,
it requires a step where probabilities are explicitly computed. Hence, we provide
`ReverseDispersion` as a probability estimator.

See [`entropy_reverse_dispersion`](@ref) for the meaning of parameters.

!!! info
    This estimator is only available for probability estimation.

[^Li2019]: Li, Y., Gao, X., & Wang, L. (2019). Reverse dispersion entropy: a new
    complexity measure for sensor signal. Sensors, 19(23), 5203.
"""
Base.@kwdef struct ReverseDispersion <: ProbabilitiesEstimator
    s = GaussianSymbolization(n_categories = 5)
    m = 2
    τ = 1
    check_unique = false
end

function distance_to_whitenoise(𝐩::Probabilities, N, m)
    # We can safely skip non-occurring symbols, because they don't contribute
    # to the sum in eq. 3 in Li et al. (2019)
    return sum(𝐩[i]^2 for i in eachindex(𝐩)) - 1/(N^m)
end

function probabilities(x::AbstractVector, est::ReverseDispersion)
    if est.check_unique
        if length(unique(x)) == 1
            symbols = repeat([1], length(x))
        else
            symbols = symbolize(x, est.s)
        end
    else
        symbols = symbolize(x, est.s)
    end
    m, τ = est.m, est.τ
    τs = tuple((x for x in 0:-τ:-(m-1)*τ)...)
    dispersion_patterns = genembed(symbols, τs, ones(m))
    N = length(x)
    𝐩 = Probabilities(dispersion_histogram(dispersion_patterns, N, est.m, est.τ))
end

"""
    entropy_reverse_dispersion(x::AbstractVector;
        s = GaussianSymbolization(n_categories = 5),
        m = 2, τ = 1, normalize = true, check_unique = true)

Estimate reverse dispersion entropy (Li et al., 2019)[^Li2019].

Relative frequencies of dispersion patterns are computed using
the symbolization scheme `s` with embedding dimension `m` and embedding delay `τ`.
Recommended parameter values[^Li2018] are `m ∈ [2, 3]`, `τ = 1`, and
`n_categories ∈ [3, 4, …, 8]` for the Gaussian mapping (defaults to 5).
The total number of possible symbols is `n_categories^m`.
If `normalize == true`, then normalize to `[0, 1]`.

## Input data

The input must have more than one unique element for the Gaussian mapping to be
well-defined. If `check_unique == true` (default), then it is checked that the input has
more than one unique value. If `check_unique == false` and the input only has one
unique element, then a `InexactError` is thrown.

See also: [`ReverseDispersion`](@ref).

[^Li2019]: Li, Y., Gao, X., & Wang, L. (2019). Reverse dispersion entropy: a new
    complexity measure for sensor signal. Sensors, 19(23), 5203.
"""
function entropy_reverse_dispersion(x::AbstractVector{T};
        s = GaussianSymbolization(n_categories = 5),
        m = 2, τ = 1, normalize = true,
        check_unique = true) where T <: Real

    est = ReverseDispersion(s = s, m = m, τ = τ, check_unique = check_unique)
    𝐩 = probabilities(x, est)
    Hrde = distance_to_whitenoise(𝐩, s.n_categories, m)

    if normalize
        # The factor `f` considers *all* possible symbols (also non-occurring)
        f = s.n_categories^m
        return Hrde / (1 - (1/f))
    else
        return Hrde
    end
end
