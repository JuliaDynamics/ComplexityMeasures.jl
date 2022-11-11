import DelayEmbeddings
export Dispersion

"""
    Dispersion(; c = 5, m = 2, τ = 1, check_unique = true)

A probability estimator based on dispersion patterns, originally used by
Rostaghi & Azami, 2016[^Rostaghi2016] to compute the "dispersion entropy", which
characterizes the complexity and irregularity of a time series.

Recommended parameter
values[^Li2018] are `m ∈ [2, 3]`, `τ = 1` for the embedding, and `c ∈ [3, 4, …, 8]`
categories for the Gaussian symbol mapping.

## Description

Assume we have a univariate time series ``X = \\{x_i\\}_{i=1}^N``. First, this time series
is discretized a "Gaussian encoding", which uses the normal cumulative distribution
function (CDF) to encode a timeseries ``x`` as integers like so:
each ``x_i`` to a new real number ``y_i \\in [0, 1]`` by using the normal
cumulative distribution function (CDF), ``x_i \\to y_i : y_i = \\dfrac{1}{ \\sigma
\\sqrt{2 \\pi}} \\int_{-\\infty}^{x_i} e^{(-(x_i - \\mu)^2)/(2 \\sigma^2)} dx``,
where ``\\mu`` and ``\\sigma`` are the empirical mean and standard deviation of ``X``.
Other choices of CDFs are also possible, but currently only Gaussian is implemented.
Next, each ``y_i`` is linearly mapped to an integer
``z_i \\in [1, 2, \\ldots, c]`` using the map
``y_i \\to z_i : z_i = R(y_i(c-1) + 0.5)``, where ``R`` indicates rounding up to the
nearest integer. This procedure subdivides the interval ``[0, 1]`` into ``c``
different subintervals that form a covering of ``[0, 1]``, and assigns each ``y_i`` to one
of these subintervals. The original time series ``X`` is thus transformed to a symbol time
series ``S = \\{ s_i \\}_{i=1}^N``, where ``s_i \\in [1, 2, \\ldots, c]``.

Next, the symbol time series ``S`` is embedded into an
``m``-dimensional time series, using an embedding lag of ``\\tau = 1``, which yields a
total of ``N - (m - 1)\\tau`` points, or "dispersion patterns". Because each ``z_i`` can
take on ``c`` different values, and each embedding point has ``m`` values, there
are ``c^m`` possible dispersion patterns. This number is used for normalization when
computing dispersion entropy.

### Computing dispersion probabilities and entropy

A probability distribution ``P = \\{p_i \\}_{i=1}^{c^m}``, where
``\\sum_i^{c^m} p_i = 1``, can then be estimated by counting and sum-normalising
the distribution of dispersion patterns among the embedding vectors.
Note that dispersion patterns that are not present are not counted. Therefore, you'll
always get non-zero probabilities using the `Dispersion` probability estimator.

## Outcome space
The outcome space for `Dispersion` is the unique delay vectos with elements the
the symbols (integers) encoded by the Gaussian PDF.
Hence, the outcome space is all `m`-dimensional delay vectors whose elements
are all possible values in `1:c`.

## Data requirements and parameters

The input must have more than one unique element for the Gaussian mapping to be
well-defined. Li et al. (2018) recommends that `x` has at least 1000 data points.

If `check_unique == true` (default), then it is checked that the input has
more than one unique value. If `check_unique == false` and the input only has one
unique element, then a `InexactError` is thrown when trying to compute probabilities.

!!! note "Why 'dispersion patterns'?"
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
struct Dispersion{S <: Encoding} <: ProbabilitiesEstimator
    encoding::S
    m::Int
    τ::Int
    check_unique::Bool
end

function Dispersion(; c = 5, m = 2, τ = 1, check_unique = false)
    return Dispersion(GaussianCDFEncoding(c = c), m, τ, check_unique)
end

function dispersion_histogram(x::AbstractDataset, N, m, τ)
    return fasthist!(x) ./ (N - (m - 1)*τ)
end

# A helper function that makes sure the algorithm doesn't crash when input contains
# a singular value.
function symbolize_for_dispersion(x, est::Dispersion)
    if est.check_unique
        if length(unique(x)) == 1
            symbols = repeat([1], length(x))
        else
            symbols = outcomes(x, est.encoding)
        end
    else
        symbols = outcomes(x, est.encoding)
    end

    return symbols::Vector{Int}
end

function probabilities_and_outcomes(x::AbstractVector{<:Real}, est::Dispersion)
    N = length(x)
    symbols = symbolize_for_dispersion(x, est)
    # We must use genembed, not embed, to make sure the zero lag is included
    m, τ = est.m, est.τ
    τs = tuple((x for x in 0:-τ:-(m-1)*τ)...)
    dispersion_patterns = genembed(symbols, τs, ones(m))
    hist = dispersion_histogram(dispersion_patterns, N, est.m, est.τ)
    return Probabilities(hist), dispersion_patterns
end

total_outcomes(est::Dispersion)::Int = est.encoding.c ^ est.m

function outcome_space(est::Dispersion)
    combs = Combinatorics.with_replacement_combinations(1:est.c, est.m)
    Ω = map(v -> SVector{est.m, Int}(v), combs)
    return Ω
end