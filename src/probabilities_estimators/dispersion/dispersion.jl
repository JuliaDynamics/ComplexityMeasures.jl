using DelayEmbeddings
using StatsBase

export Dispersion

"""
    Dispersion(; s = GaussianSymbolization(5), m = 2, τ = 1, check_unique = true,
        normalize = true)

A probability estimator based on dispersion patterns, originally used by
Rostaghi & Azami, 2016[^Rostaghi2016] to compute the "dispersion entropy", which
characterizes the complexity and irregularity of a time series

Relative frequencies of dispersion patterns are computed using the symbolization scheme
`s` with embedding dimension `m` and embedding delay `τ`. Recommended parameter
values[^Li2018] are `m ∈ [2, 3]`, `τ = 1`, and `n_categories ∈ [3, 4, …, 8]` for the
Gaussian symbol mapping (defaults to 5).

If `normalize == true`, then when used in combination with [`entropy_renyi`](@ref)
(see below), the the dispersion entropy is normalized to `[0, 1]`. Normalization is only
defined when `q == 1`.

If `check_unique == true` (default), then it is checked that the input has
more than one unique value. If `check_unique == false` and the input only has one
unique element, then a `InexactError` is thrown when trying to compute probabilities.

## Probabilities vs dispersion entropy

The original dispersion entropy paper does not discuss the technique as a probability
estimator per se, but does require a step where probabilities over dispersion patterns
are explicitly computed. Hence, we provide `Dispersion` as a probability estimator.

To compute (normalized) dispersion entropy of order `q` to a given `base` on the
univariate input time series `x`, do:

```julia
renyi_entropy(x, Dispersion(normalize = true), base = base, q = q)
```

## Data requirements

The input must have more than one unique element for the Gaussian mapping to be
well-defined. Li et al. (2018) recommends that `x` has at least 1000 data points.

## Further reading

See the [online examples](@ref dispersion_examples) for a more in-depth description of
dispersion pattern estimation and usage.

[^Rostaghi2016]: Rostaghi, M., & Azami, H. (2016). Dispersion entropy: A measure for time-series analysis. IEEE Signal Processing Letters, 23(5), 610-614.
[^Li2018]: Li, G., Guan, Q., & Yang, H. (2018). Noise reduction method of underwater acoustic signals based on CEEMDAN, effort-to-compress complexity, refined composite multiscale dispersion entropy and wavelet threshold denoising. Entropy, 21(1), 11.
"""
Base.@kwdef struct Dispersion <: ProbabilitiesEstimator
    s = GaussianSymbolization(n_categories = 5)
    m = 2
    τ = 1
    check_unique = false
    normalize = true
end

export entropy_dispersion

"""
    embed_symbols(s::AbstractVector{T}, m, τ) {where T} → Dataset{m, T}

From the symbols `sᵢ ∈ s`, create the embedding vectors (with dimension `m` and lag `τ`):

```math
s_i^D = \\{s_i, s_{i+\\tau}, \\ldots, s_{i+(m-1)\\tau} \\}
```,

where ``i = 1, 2, \\ldots, N - (m - 1)\\tau`` and `N = length(s)`.
"""
function embed_symbols(symbols::AbstractVector, m, τ)
    return embed(symbols, m, τ)
end

function dispersion_histogram(x::AbstractDataset, N, m, τ)
    return _non0hist(x.data, (N - (m - 1)*τ))
end

function probabilities(x::AbstractVector, est::Dispersion)
    if est.check_unique
        if length(unique(x)) == 1
            symbols = repeat([1], length(x))
        else
            symbols = symbolize(x, est.s)
        end
    else
        symbols = symbolize(x, est.s)
    end
    N = length(x)

    # We must use genembed, not embed, to make sure the zero lag is included
    m, τ = est.m, est.τ
    τs = tuple((x for x in 0:-τ:-(m-1)*τ)...)
    dispersion_patterns = genembed(symbols, τs, ones(m))
    hist = dispersion_histogram(dispersion_patterns, N, est.m, est.τ)
    p = Probabilities(hist)
end

function entropy_renyi(x::AbstractVector, est::Dispersion; q = 1, base = MathConstants.e)
    p = probabilities(x, est)
    dh = entropy_renyi(p, q = q, base = base)

    n, m = est.s.n_categories, est.m

    if est.normalize
        # TODO: is is possible to normalize for general order `q`? Need to have a literature
        # dive or figure it out manually.
        if q == 1
            return dh / log(base, n^m)
        else
            throw(ArgumentError("Normalization is not well defined when q != 1."))
        end
    else
        return dh
    end
end
