export reverse_dispersion

function distance_to_whitenoise(p::Probabilities, n_classes, m; normalize = false)
    # We can safely skip non-occurring symbols, because they don't contribute
    # to the sum in eq. 3 in Li et al. (2019)
    Hrde = sum(abs2, p) - 1/(n_classes^m)

    if normalize
        # The factor `f` considers *all* possible symbols (also non-occurring)
        f = n_classes^m
        return Hrde / (1 - (1/f))
    else
        return Hrde
    end
end

"""
    reverse_dispersion(x::AbstractVector{T}, s = GaussianSymbolization(c = 5), m = 2, τ = 1,
        normalize = true)


Compute the reverse dispersion entropy complexity measure (Li et al., 2019)[^Li2019].

## Algorithm

Li et al. (2021)[^Li2019] defines the reverse dispersion entropy as

```math
H_{rde} = \\sum_{i = 1}^{c^m} \\left(p_i - \\dfrac{1}{{c^m}} \\right)^2.
```

where the probabilities ``p_i`` are obtained precisely as for the [`Dispersion`](@ref)
probability estimator. Relative frequencies of dispersion patterns are computed using the
symbolization scheme `s`, which defaults to symbolization using the normal cumulative
distribution function (NCDF), as implemented by [`GaussianSymbolization`](@ref), using
embedding dimension `m` and embedding delay `τ`.
Recommended parameter values[^Li2018] are `m ∈ [2, 3]`, `τ = 1` for the embedding, and
`c ∈ [3, 4, …, 8]` categories for the Gaussian mapping. If `normalize == true`, then
the reverse dispersion entropy is normalized to `[0, 1]`.

The minimum value of ``H_{rde}`` is zero and occurs precisely when the dispersion
pattern distribution is flat, which occurs when all ``p_i``s are equal to ``1/c^m``.
Because ``H_{rde} \\geq 0``, ``H_{rde}`` can therefore be said to be a measure of how far
the dispersion pattern probability distribution is from white noise.

## Example

```jldoctest reverse_dispersion_example; setup = :(using Entropies)
julia> x = repeat([0.5, 0.7, 0.1, -1.0, 1.11, 2.22, 4.4, 0.2, 0.2, 0.1], 10);

julia> c, m = 3, 5;

julia> reverse_dispersion(x, s = GaussianSymbolization(c = c), m = m, normalize = false)
0.11372331532921814
```

!!! note
    #### A clarification on notation

    With ambiguous notation, Li et al. claim that

    ``H_{rde} = \\sum_{i = 1}^{c^m} \\left(p_i - \\dfrac{1}{{c^m}} \\right)^2 = \\sum_{i = 1}^{c^m} p_i^2 - \\frac{1}{c^m}.``

    But on the right-hand side of the equality, does the constant term appear within or
    outside the sum? Using that ``P`` is a probability distribution by
    construction (in step 4) , we see that the constant must appear *outside* the sum:

    ```math
    \\begin{aligned}
    H_{rde} &= \\sum_{i = 1}^{c^m} \\left(p_i - \\dfrac{1}{{c^m}} \\right)^2
    = \\sum_{i=1}^{c^m} p_i^2 - \\frac{2p_i}{c^m} + \\frac{1}{c^{2m}} \\\\
    &= \\left( \\sum_{i=1}^{c^m} p_i^2 \\right) - \\left(\\sum_i^{c^m} \\frac{2p_i}{c^m}\\right) + \\left( \\sum_{i=1}^{c^m} \\dfrac{1}{{c^{2m}}} \\right) \\\\
    &= \\left( \\sum_{i=1}^{c^m} p_i^2 \\right) - \\left(\\frac{2}{c^m} \\sum_{i=1}^{c^m} p_i \\right) +  \\dfrac{c^m}{c^{2m}} \\\\
    &= \\left( \\sum_{i=1}^{c^m} p_i^2 \\right) - \\frac{2}{c^m} (1) +  \\dfrac{1}{c^{m}} \\\\
    &= \\left( \\sum_{i=1}^{c^m} p_i^2 \\right) - \\dfrac{1}{c^{m}}.
    \\end{aligned}
    ```

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
    Hrde = distance_to_whitenoise(p, s.c, m)
end
