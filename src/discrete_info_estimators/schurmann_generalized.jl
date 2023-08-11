export GeneralizedSchürmann

"""
    GeneralizedSchürmann <: DiscreteInfoEstimator
    GeneralizedSchürmann(measure::Shannon; a::Union{<:Real, Vector{<:Real}} = 1.0)

The `GeneralizedSchürmann` estimator is used with [`information`](@ref) to compute the
discrete [`Shannon`](@ref) entropy with the bias-corrected estimator
given in Grassberger (2022)[^Grassberger2022].

The "generalized" part of the name, as opposed to the [`Schürmann2004`](@ref) estimator,
is due to the possibility of picking difference parameters ``a_i`` for different outcomes.
If different parameters are assigned to the different outcomes, `a` must be a vector of
parameters of length `length(outcomes)`, where the outcomes are obtained using
[`outcomes`](@ref). See Grassberger (2022) for more information. If `a` is a real number,
then ``a_i = a \\forall i``, and the estimator reduces to the [`Schürmann`](@ref) estimator.

## Description

For a set of ``N`` observations over ``M`` outcomes, the estimator is given by

```math
H_S^{opt} = \\varphi(N) - \\dfrac{1}{N} \\sum_{i=1}^M n_i G_{n_i}(a_i),
```

where ``n_i`` is the observed frequency of the i-th outcome,

```math
G_n(a) = \\varphi(n) + (-1)^n \\int_0^a \\dfrac{x^{n - 1}}{x + 1} dx,
```

``G_n(1) = G_n`` and ``G_n(0) = \\varphi(n)``, and

```math
G_n = \\varphi(n) + (-1)^n \\int_0^1 \\dfrac{x^{n - 1}}{x + 1} dx.
```

[^Grassberger2022]:
    Grassberger, P. (2022). On generalized Schürmann entropy estimators. Entropy, 24(5),
    680.
"""
Base.@kwdef struct GeneralizedSchürmann{I <: InformationMeasure, T} <: DiscreteInfoEstimator{I}
    definition::I = Shannon()
    # `a[i]` is the parameter for the i-th outcome, and there must be one
    # parameter per outcome. The user should construct
    # `a[i]` by calling `outcomes` with the desired probabilities estimator (with input
    # data, if necessary), and assign one parameter to each outcome.
    a::Union{T, Vector{T}} = 1.0
end
function GeneralizedSchürmann(definition::I; a::A = 1.0) where {I, A}
    all(a .> 0) || throw(ArgumentError("All elements of `a` must be strict positive. Got $a."))
    return GeneralizedSchürmann(; definition, a)
end
function information(hest::GeneralizedSchürmann{<:Shannon}, pest::ProbabilitiesEstimator, x)
    (; definition, a) = hest

    freqs = counts(pest, x)
    # We should be using `N = length(x)`, but since some probabilities estimators
    # return pseudo counts, we need to consider those instead of counting actual
    # observations.
    N = sum(freqs)

    if a isa Real
        h = digamma(N) - 1/N * sum(nᵢ * Gₙ(a, nᵢ) for nᵢ in freqs)
    else
        # Assumes a[i] corresponds to freqs[i]
        h = digamma(N) - 1/N * sum(nᵢ * Gₙ(aᵢ, nᵢ) for (nᵢ, aᵢ) in zip(freqs, a))
    end

    # Grassberger's estimate of `h` is based on the natural logarithm, so we must convert
    # to the desired base.
    return convert_logunit(h, MathConstants.e, definition.base)
end

function Gₙ(a, n::Int)
    if a == 0.0
        return digamma(n)
    elseif a == 1.0
        return Gₙ(n)
    end

    # Integrate `f_schurmann` from `lb` to `ub`. We only need the value of the integral
    # (not the error), so index first element returned from `quadgk`
    lb = 0.0
    ub = a
    integ = first(quadgk(x -> f_schurmann(x, n), lb, ub))

    return digamma(n) + (-1)^n * integ
end


function Gₙ(n::Int)
    if n == 0
        return 0.0
    end
    integ = first(quadgk(x -> f_schurmann(x, n), 0.0, 1.0))
    return digamma(n) + (-1)^n * integ
end
