using Statistics
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
is encoded into a symbol timeseries ``S`` using the Gaussian encoding
[`GaussianCDFEncoding`](@ref) with empirical mean `μ` and empirical standard deviation `σ`
(both determined from ``X``), and `c` as given to `Dispersion`.

Then, ``S`` is embedded into an ``m``-dimensional time series, using an embedding lag of
``\\tau``, which yields a total of ``N - (m - 1)\\tau`` delay vectors ``z_i``,
or "dispersion patterns". Since each element of ``z_i`` can take on `c` different values,
and each delay vector has `m` entries, there are `c^m` possible dispersion patterns.
This number is used for normalization when computing dispersion entropy.

The returned probabilities are simply the frequencies of the unique dispersion patterns
present in ``S`` (i.e., the [`CountOccurences`](@ref) of ``S``).

## Outcome space

The outcome space for `Dispersion` is the unique delay vectors whose elements are the
the symbols (integers) encoded by the Gaussian CDF, i.e., the unique elements of ``S``.

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
    close together (close to the mean), so it represents a set of numbers that
    are not very spread out (less dispersed). The embedding vector ``[1, 1, 2, 3, 3]``,
    however, represents numbers that are much more spread out (more dispersed), because the
    categories representing "outliers" both above and below the mean are represented,
    not only values close to the mean.

For a version of this estimator that can be used on high-dimensional arrays, see
[`SpatialDispersion`](@ref).

[^Rostaghi2016]:
    Rostaghi, M., & Azami, H. (2016). Dispersion entropy: A measure for time-series analysis.
    IEEE Signal Processing Letters, 23(5), 610-614.

[^Li2018]:
    Li, G., Guan, Q., & Yang, H. (2018). Noise reduction method of underwater acoustic
    signals based on CEEMDAN, effort-to-compress complexity, refined composite multiscale
    dispersion entropy and wavelet threshold denoising. InformationMeasure, 21(1), 11.
"""
Base.@kwdef struct Dispersion{S <: Encoding} <: OutcomeSpaceModel
    encoding::Type{S} = GaussianCDFEncoding # any encoding at accepts keyword `c`
    c::Int = 3 # The number of categories to map encoded values to.
    m::Int = 2
    τ::Int = 1
    check_unique::Bool = false
end
# TODO: the normalization should happen in `probabilities``, not here
function dispersion_histogram(x::AbstractStateSpaceSet, N, m, τ)
    return fasthist!(x) #./ (N - (m - 1)*τ)
end

# A helper function that makes sure the algorithm doesn't crash when input contains
# a singular value.
function symbolize_for_dispersion(est::Dispersion, x)
    σ = std(x)
    μ = mean(x)
    ENCODING_TYPE = est.encoding
    encoding = ENCODING_TYPE(; σ, μ, c = est.c)

    if est.check_unique
        if length(unique(x)) == 1
            symbols = repeat([1], length(x))
        else
            symbols = encode.(Ref(encoding), x)
        end
    else
        symbols = encode.(Ref(encoding), x)
    end

    return symbols::Vector{Int}
end

function frequencies_and_outcomes(est::Dispersion, x::AbstractVector{<:Real})
    N = length(x)
    symbols = symbolize_for_dispersion(est, x)
    # We must use genembed, not embed, to make sure the zero lag is included
    m, τ = est.m, est.τ
    τs = tuple((x for x in 0:-τ:-(m-1)*τ)...)
    dispersion_patterns = genembed(symbols, τs, ones(m))
    hist = dispersion_histogram(dispersion_patterns, N, est.m, est.τ)
    # `dispersion_patterns` is sorted when computing the histogram, so patterns match
    # the histogram values, but `dispersion_patterns` still contains repeated values,
    # so we return the unique values.
    return hist, unique(dispersion_patterns.data)
end

function outcome_space(est::Dispersion)
    c, m = 1:est.c, est.m
    cart = CartesianIndices(ntuple(i -> c, m))
    V = SVector{m, Int}
    return map(i -> V(Tuple(i)), vec(cart))
end
# Performance extension
total_outcomes(est::Dispersion)::Int = est.c ^ est.m
