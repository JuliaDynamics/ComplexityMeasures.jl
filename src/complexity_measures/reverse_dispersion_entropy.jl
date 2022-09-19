export reverse_dispersion

function distance_to_whitenoise(p::Probabilities, N, m)
    # We can safely skip non-occurring symbols, because they don't contribute
    # to the sum in eq. 3 in Li et al. (2019)
    return sum(abs2, p) - 1/(N^m)
end

"""
    reverse_dispersion(x::AbstractVector{T}, s = GaussianSymbolization(5), m = 2, τ = 1,
        normalize = true)

Compute the reverse dispersion entropy complexity measure (Li et al., 2019)[^Li2019],
which measures how far from being white noise a signal is.

Like for [`Dispersion`](@ref), relative frequencies of dispersion patterns are computed
using the symbolization scheme `s` with embedding dimension `m` and embedding delay `τ`.
Recommended parameter values[^Li2018] are `m ∈ [2, 3]`, `τ = 1`, and
`n_categories ∈ [3, 4, …, 8]` for the Gaussian mapping (defaults to 5).
If `normalize == true`, then the reverse dispersion entropy is normalized to `[0, 1]`.

[^Li2019]: Li, Y., Gao, X., & Wang, L. (2019). Reverse dispersion entropy: a new
    complexity measure for sensor signal. Sensors, 19(23), 5203.
"""
function reverse_dispersion(x::AbstractVector{T}; s = GaussianSymbolization(5),
        m = 2, τ = 1, normalize = true) where T <: Real
    est = Dispersion(τ = τ, m = m, s = s)
    p = probabilities(x, est)

    # The following step combines distance information with the probabilities, so
    # from here on, it is not possible to use `renyi_entropy` or similar methods, because
    # we're not dealing with probabilities anymore.
    Hrde = distance_to_whitenoise(p, s.n_categories, m)

    if normalize
        # The factor `f` considers *all* possible symbols (also non-occurring)
        f = s.n_categories^m
        return Hrde / (1 - (1/f))
    else
        return Hrde
    end
end
