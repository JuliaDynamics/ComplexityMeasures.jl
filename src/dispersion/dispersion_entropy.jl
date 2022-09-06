using DelayEmbeddings
using StatsBase

export dispersion_entropy

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


"""
    dispersion_entropy(x, s = GaussianSymbolization(n_categories = 5);
        m = 3, τ = 1, q = 1, base = MathConstants.e)

Compute the (order-`q` generalized) dispersion entropy to the given `base` of the
univariate time series `x`. Relative frequencies of dispersion patterns are computed using
the symbolization scheme `s` with embedding dimension `m` and embedding delay `τ`.

Recommended parameter values[^Li2018] are `m ∈ [2, 3]`, `τ = 1`, and
`n_categories ∈ [3, 4, …, 8]` for the Gaussian mapping (defaults to 5).

## Description

Dispersion entropy characterizes the complexity and irregularity of a time series.
This implementation follows the description in Li et al. (2018)[^Li2018], which is
based on Azami & Escudero (2018)[^Azami2018], additionally allowing the computation of
generalized dispersion entropy of order `q` (default is `q = 1`, which is the Shannon entropy).

## Data requirements

Li et al. (2018) recommends that `x` has at least 1000 data points.

[^Li2018]: Li, G., Guan, Q., & Yang, H. (2018). Noise reduction method of underwater acoustic signals based on CEEMDAN, effort-to-compress complexity, refined composite multiscale dispersion entropy and wavelet threshold denoising. Entropy, 21(1), 11.
[^Azami2018]: Azami, H., & Escudero, J. (2018). Coarse-graining approaches in univariate multiscale sample and dispersion entropy. Entropy, 20(2), 138.
"""
function dispersion_entropy(x::AbstractVector,
        s::GaussianSymbolization = GaussianSymbolization(n_categories = 5);
        m = 2, τ = 1, q = 1, base = MathConstants.e)
    symbols = symbolize(x, s)
    dispersion_patterns = embed(symbols, m, τ)
    N = length(x)

    hist = dispersion_histogram(dispersion_patterns, N, m, τ)
    entropy_renyi(hist, q = q, base = base)
end
