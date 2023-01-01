using ComplexityMeasures: Dispersion, Encoding, Probabilities, GaussianCDFEncoding
import ComplexityMeasures: total_outcomes

export ReverseDispersion

"""
    ReverseDispersion <: ComplexityMeasure
    ReverseDispersion(; c = 3, m = 2, τ = 1, check_unique = true)

Estimator for the reverse dispersion entropy complexity measure (Li et al., 2019)[^Li2019].

## Description

Li et al. (2021)[^Li2019] defines the reverse dispersion entropy as

```math
H_{rde} = \\sum_{i = 1}^{c^m} \\left(p_i - \\dfrac{1}{{c^m}} \\right)^2 =
\\left( \\sum_{i=1}^{c^m} p_i^2 \\right) - \\dfrac{1}{c^{m}}
```
where the probabilities ``p_i`` are obtained precisely as for the [`Dispersion`](@ref)
probability estimator. Relative frequencies of dispersion patterns are computed using the
given `encoding` scheme , which defaults to encoding using the normal cumulative
distribution function (NCDF), as implemented by [`GaussianCDFEncoding`](@ref), using
embedding dimension `m` and embedding delay `τ`.
Recommended parameter values[^Li2018] are `m ∈ [2, 3]`, `τ = 1` for the embedding, and
`c ∈ [3, 4, …, 8]` categories for the Gaussian mapping.

If normalizing, then the reverse dispersion entropy is normalized to `[0, 1]`.

The minimum value of ``H_{rde}`` is zero and occurs precisely when the dispersion
pattern distribution is flat, which occurs when all ``p_i``s are equal to ``1/c^m``.
Because ``H_{rde} \\geq 0``, ``H_{rde}`` can therefore be said to be a measure of how far
the dispersion pattern probability distribution is from white noise.

## Data requirements

The input must have more than one unique element for the default [`GaussianEncoding`](@ref)
to be well-defined. Li et al. (2018) recommends
that `x` has at least 1000 data points.

If `check_unique == true` (default), then it is checked that the input has
more than one unique value. If `check_unique == false` and the input only has one
unique element, then a `InexactError` is thrown when trying to compute probabilities.

[^Li2019]: Li, Y., Gao, X., & Wang, L. (2019). Reverse dispersion entropy: a new
    complexity measure for sensor signal. Sensors, 19(23), 5203.
"""
Base.@kwdef struct ReverseDispersion{S <: Encoding} <: ComplexityMeasure
    encoding::Type{S} = GaussianCDFEncoding # any encoding at accepts keyword `c`
    c::Int = 3 # The number of categories to map encoded values to.
    m::Int = 2
    τ::Int = 1
    check_unique::Bool = false
end

total_outcomes(est::ReverseDispersion)::Int = est.c ^ est.m

"""
    distance_to_whitenoise(estimator::ReverseDispersion, p::Probabilities;
        normalize = false)

Compute the distance of the probability distribution `p` from a uniform distribution,
given the parameters of `estimator` (which must be known beforehand).

If `normalize == true`, then normalize the value to the interval `[0, 1]` by using the
parameters of `estimator`.

Used to compute reverse dispersion entropy([`ReverseDispersion`](@ref);
Li et al., 2019[^Li2019]).

[^Li2019]: Li, Y., Gao, X., & Wang, L. (2019). Reverse dispersion entropy: a new
    complexity measure for sensor signal. Sensors, 19(23), 5203.
"""
function distance_to_whitenoise(est::ReverseDispersion, p::Probabilities; normalize = false)
    # We can safely skip non-occurring symbols, because they don't contribute
    # to the sum in eq. 3 in Li et al. (2019)
    Hrde = sum(abs2, p) - (1 / total_outcomes(est))

    if normalize
        return Hrde / (1 - (1 / total_outcomes(est)))
    else
        return Hrde
    end
end

function complexity(measure::ReverseDispersion, x)
    (; encoding, c, m, τ, check_unique) = measure
    p = probabilities(Dispersion(; c, m, τ, check_unique), x)
    return distance_to_whitenoise(measure, p, normalize = false)
end

function complexity_normalized(measure::ReverseDispersion, x)
    (; encoding, c, m, τ, check_unique) = measure
    p = probabilities(Dispersion(; c, m, τ, check_unique), x)
    return distance_to_whitenoise(measure, p, normalize = true)
end
