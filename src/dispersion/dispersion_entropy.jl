using DelayEmbeddings
using StatsBase

export Dispersion
export entropy_dispersion
include("GaussianSymbolization.jl")

"""
    Dispersion(; s = GaussianSymbolization(5), m = 2, τ = 1, check_unique = true)

A probability estimator using the dispersion entropy technique from
Rostaghi & Azami (2016)[^Rostaghi2016].

Although the dispersion entropy is not intended as a probability estimator per se,
it requires a step where probabilities are explicitly computed. Hence, we provide
`Dispersion` as a probability estimator.

See [`entropy_dispersion`](@ref) for the meaning of parameters.

!!! info
    This estimator is only available for probability estimation.

[^Rostaghi2016]: Rostaghi, M., & Azami, H. (2016). Dispersion entropy: A measure for time-series analysis. IEEE Signal Processing Letters, 23(5), 610-614.
"""
Base.@kwdef struct Dispersion <: ProbabilitiesEstimator
    s = GaussianSymbolization(n_categories = 5)
    m = 2
    τ = 1
    check_unique = false
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

    𝐩 = Probabilities(dispersion_histogram(dispersion_patterns, N, est.m, est.τ))
end

"""
    entropy_dispersion(x, s = GaussianSymbolization(n_categories = 5);
        m = 3, τ = 1, q = 1, base = MathConstants.e)

Compute the dispersion entropy (Rostaghi & Azami, 2016)[^Rostaghi2016] to order `q`
and the given `base` of the univariate time series `x`. Relative frequencies of dispersion
patterns are computed using the symbolization scheme `s` with embedding dimension `m` and
embedding delay `τ`.

Recommended parameter values[^Li2018] are `m ∈ [2, 3]`, `τ = 1`, and
`n_categories ∈ [3, 4, …, 8]` for the Gaussian mapping (defaults to 5).

## Description

Dispersion entropy characterizes the complexity and irregularity of a time series.
This implementation follows the description in Li et al. (2018)[^Li2018], which is
based on Azami & Escudero (2018)[^Azami2018], additionally allowing the computation of
generalized dispersion entropy of order `q` (default is `q = 1`, which is the Shannon entropy).

## Data requirements

The input must have more than one unique element for the Gaussian mapping to be
well-defined. If `check_unique == true` (default), then it is checked that the input has
more than one unique value. If `check_unique == false` and the input only has one
unique element, then a `InexactError` is thrown.

Li et al. (2018) recommends that `x` has at least 1000 data points.

See also: [`Dispersion`](@ref).

[^Rostaghi2016]: Rostaghi, M., & Azami, H. (2016). Dispersion entropy: A measure for time-series analysis. IEEE Signal Processing Letters, 23(5), 610-614.
[^Li2018]: Li, G., Guan, Q., & Yang, H. (2018). Noise reduction method of underwater acoustic signals based on CEEMDAN, effort-to-compress complexity, refined composite multiscale dispersion entropy and wavelet threshold denoising. Entropy, 21(1), 11.
[^Azami2018]: Azami, H., & Escudero, J. (2018). Coarse-graining approaches in univariate multiscale sample and dispersion entropy. Entropy, 20(2), 138.
"""
function entropy_dispersion(x::AbstractVector;
        s::GaussianSymbolization = GaussianSymbolization(n_categories = 5),
        m = 2, τ = 1, q = 1, base = MathConstants.e, normalize = true,
        check_unique = true)
    est = Dispersion(m = m, τ = τ, s = s, check_unique = check_unique)
    𝐩 = probabilities(x, est)

    dh = genentropy(𝐩, q = q, base = base)

    if normalize
        return dh / log(base, s.n_categories^m)
    else
        return dh
    end
end
