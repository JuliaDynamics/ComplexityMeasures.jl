using DelayEmbeddings
using StatsBase

export Dispersion

"""
    Dispersion(; s = GaussianSymbolization(c = 5), m = 2, τ = 1, check_unique = true,
        normalize = true)

A probability estimator based on dispersion patterns, originally used by
Rostaghi & Azami, 2016[^Rostaghi2016] to compute the "dispersion entropy", which
characterizes the complexity and irregularity of a time series.

Relative frequencies of dispersion patterns are computed using the symbolization scheme
`s` with embedding dimension `m` and embedding delay `τ`. Recommended parameter
values[^Li2018] are `m ∈ [2, 3]`, `τ = 1` for the embedding, and `c ∈ [3, 4, …, 8]`
categories for the Gaussian symbol mapping. See below for an in-depth description of
dispersion pattern construction and usage.

# Algorithm

Assume we have a univariate time series ``X = \\{x_i\\}_{i=1}^N``. First, this time series
is symbolized using `s`, which default to [`GaussianSymbolization`](@ref), which uses the
normal cumulative distribution function (CDF) for symbolization. Other choices of CDFs are
also possible, but Entropies.jl currently only implements [`GaussianSymbolization`](@ref),
which was used in Rostaghi & Azami (2016). This step results in an integer-valued symbol
time series ``S = \\{ s_i \\}_{i=1}^N``, where ``s_i \\in [1, 2, \\ldots, c]``.

Next, the symbol time series ``S`` is embedded into an
``m``-dimensional time series, using an embedding lag of ``\\tau = 1``, which yields a
total of ``N - (m - 1)\\tau`` points, or "dispersion patterns". Because each ``z_i`` can
take on ``c`` different values, and each embedding point has ``m`` values, there
are ``c^m`` possible dispersion patterns. This number is used for normalization when
computing dispersion entropy.

## Computing dispersion probabilities and entropy

A probability distribution ``P = \\{p_i \\}_{i=1}^{c^m}``, where
``\\sum_i^{c^m} p_i = 1``, can then be estimated by counting and sum-normalising
the distribution of dispersion patterns among the embedding vectors.

```jldoctest dispersion_example; setup = :(using Entropies)
julia> x = repeat([0.5, 0.7, 0.1, -1.0, 1.11, 2.22, 4.4, 0.2, 0.2, 0.1], 10);

julia> c, m = 3, 5;

julia> est = Dispersion(s = GaussianSymbolization(c = c), m = m);

julia> probs = probabilities(x, est)
9-element Probabilities{Float64}:
 0.09375000000000001
 0.10416666666666669
 0.18750000000000003
 0.09375000000000001
 0.10416666666666669
 0.10416666666666669
 0.10416666666666669
 0.10416666666666669
 0.10416666666666669
```

Note that dispersion patterns that are not present are not counted. Therefore, you'll
always get non-zero probabilities using the `Dispersion` probability estimator.

To compute (normalized) dispersion entropy of order `q` to a given `base` on the
univariate input time series `x`, do:

```jldoctest dispersion_example
julia> entropy_renyi(x, Dispersion(normalize = true), base = 2, q = 1)
0.6716889280681666
```

If `normalize == true`, then when used in combination with [`entropy_renyi`](@ref),
the dispersion entropy is normalized to `[0, 1]`. Normalization is only
defined when `q == 1`.

## Data requirements and parameters

The input must have more than one unique element for the Gaussian mapping to be
well-defined. Li et al. (2018) recommends that `x` has at least 1000 data points.

If `check_unique == true` (default), then it is checked that the input has
more than one unique value. If `check_unique == false` and the input only has one
unique element, then a `InexactError` is thrown when trying to compute probabilities.

See also: [`entropy_dispersion`](@ref), [`GaussianSymbolization`](@ref).


!!! note
    ## Why "dispersion patterns"?
    Each embedding vector is called a "dispersion pattern". Why? Let's consider the case
    when ``m = 5`` and ``c = 3``, and use some very imprecise terminology for illustration:

    When ``c = 3``, values clustering far below mean are in one group, values clustered
    around the mean are in one group, and values clustering far above the mean are in a
    third group. Then the embedding vector ``[2, 2, 2, 2, 2]`` consists of values that are
    relatively close together (close to the mean), so it represents a set of numbers that
    are not very spread out (less dispersed). The embedding vector ``[1, 1, 2, 3, 3]``,
    however, represents numbers that are much more spread out (more dispersed), because the
    categories representing "outliers" both above and below the mean are represented,
    not only values close to the mean.

[^Rostaghi2016]: Rostaghi, M., & Azami, H. (2016). Dispersion entropy: A measure for time-series analysis. IEEE Signal Processing Letters, 23(5), 610-614.
[^Li2018]: Li, G., Guan, Q., & Yang, H. (2018). Noise reduction method of underwater acoustic signals based on CEEMDAN, effort-to-compress complexity, refined composite multiscale dispersion entropy and wavelet threshold denoising. Entropy, 21(1), 11.
"""
Base.@kwdef struct Dispersion <: ProbabilitiesEstimator
    s = GaussianSymbolization(c = 5)
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

    n, m = est.s.c, est.m

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