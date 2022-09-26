export reverse_dispersion
export distance_to_whitenoise

# Note: this is not an entropy estimator, so we don't use the entropy_xxx_norm interface
# for normalization, even though we rely on `alphabet_length`.
"""
    distance_to_whitenoise(p::Probabilities, estimator::Dispersion; normalize = false)

Compute the distance of the probability distribution `p` from a uniform distribution,
given the parameters of `estimator` (which must be known beforehand).

If `normalize == true`, then normalize the value to the interval `[0, 1]` by using the
parameters of `estimator`.

Used to compute reverse dispersion entropy([`reverse_dispersion`](@ref);
Li et al., 2019[^Li2019]).

[^Li2019]: Li, Y., Gao, X., & Wang, L. (2019). Reverse dispersion entropy: a new
    complexity measure for sensor signal. Sensors, 19(23), 5203.
"""
function distance_to_whitenoise(p::Probabilities, est::Dispersion; normalize = false)
    # We can safely skip non-occurring symbols, because they don't contribute
    # to the sum in eq. 3 in Li et al. (2019)
    Hrde = sum(abs2, p) - (1 / alphabet_length(est))

    if normalize
        return Hrde / (1 - (1 / alphabet_length(est)))
    else
        return Hrde
    end
end

# Note again: this is a *complexity measure*, not an entropy estimator, so we don't use
# the entropy_xxx_norm interface for normalization, even though we rely on `alphabet_length`.
"""
    reverse_dispersion(x::AbstractVector{T}, est::Dispersion = Dispersion();
        normalize = true) where T <: Real

Compute the reverse dispersion entropy complexity measure (Li et al., 2019)[^Li2019].

## Description

Li et al. (2021)[^Li2019] defines the reverse dispersion entropy as

```math
H_{rde} = \\sum_{i = 1}^{c^m} \\left(p_i - \\dfrac{1}{{c^m}} \\right)^2 =
\\left( \\sum_{i=1}^{c^m} p_i^2 \\right) - \\dfrac{1}{c^{m}}
```
where the probabilities ``p_i`` are obtained precisely as for the [`Dispersion`](@ref)
probability estimator. Relative frequencies of dispersion patterns are computed using the
given `symbolization` scheme , which defaults to symbolization using the normal cumulative
distribution function (NCDF), as implemented by [`GaussianSymbolization`](@ref), using
embedding dimension `m` and embedding delay `τ`.
Recommended parameter values[^Li2018] are `m ∈ [2, 3]`, `τ = 1` for the embedding, and
`c ∈ [3, 4, …, 8]` categories for the Gaussian mapping.

If `normalize == true`, then the reverse dispersion entropy is normalized to `[0, 1]`.

The minimum value of ``H_{rde}`` is zero and occurs precisely when the dispersion
pattern distribution is flat, which occurs when all ``p_i``s are equal to ``1/c^m``.
Because ``H_{rde} \\geq 0``, ``H_{rde}`` can therefore be said to be a measure of how far
the dispersion pattern probability distribution is from white noise.

[^Li2019]: Li, Y., Gao, X., & Wang, L. (2019). Reverse dispersion entropy: a new
    complexity measure for sensor signal. Sensors, 19(23), 5203.
"""
function reverse_dispersion(x::AbstractVector{T}, est::Dispersion = Dispersion();
        normalize = true) where T <: Real

    p = probabilities(x, est)

    # The following step combines distance information with the probabilities, so
    # from here on, it is not possible to use `renyi_entropy` or similar methods, because
    # we're not dealing with probabilities anymore.
    Hrde = distance_to_whitenoise(p, est, normalize = normalize)
end
