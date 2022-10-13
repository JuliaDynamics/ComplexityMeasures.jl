# This file contains an API for multiscale (coarse-grained/downsampled) computations of
# entropy and various complexity measures on time series.
# Can be used both to compute actual entropies (i.e. diversity entropy), or
# complexity measures (sample entropy, approximate entropy).

using Statistics
export multiscale
export downsample
export Regular
export Composite
export MultiScaleAlgorithm

"""
    MultiScaleAlgorithm

An abstract type for multiscale methods.
"""
abstract type MultiScaleAlgorithm end

"""
    Regular <: MultiScaleAlgorithm
    Regular(; f::Function = Statistics.mean)

The original multi-scale algorithm for multiscale entropy analysis (Costa et al.,
2022)[^Costa2002], which yields a single downsampled time series per scale `s`.

## Description

Given a scalar-valued input time series `x`, the `Regular` multiscale algorithm downsamples
and coarse-grains `x` by splitting it into non-overlapping windows of length `s`, and
then constructing a new downsampled time series `D_t(s, f)` by applying the function `f`
to each of the resulting length-`s` windows.

The downsampled time series `D_t(s)` with `t ∈ [1, 2, …, L]`, where `L = floor(N / s)`,
is given by:

```math
\\{ D_t(s, f)  \\}_{t = 1}^{L} = \\left\\{ f \\left( \\bf x_t \\right) \\right\\}_{t = 1}^{L} =
\\left\\{
    {f\\left( (x_i)_{i = (t - 1)s + 1}^{ts} \\right)}
\\right\\}_{t = 1}^{L}
```

where `f` is some summary statistic applied to the length-`ts-((t - 1)s + 1)` tuples `xₖ`.
Different choices of `f` have yield different multiscale methods appearing in the
literature. For example:

- `f == Statistics.mean` yields the original first-moment multiscale entropy (Costa
    et al., 2002)[^Costa2002].
- `f == Statistics.var` yields the "generalized multiscale entropy" (Costa & Goldberger,
    2015)[^Costa2015], which uses the second-moment (variance) instead of the mean.

[^Costa2002]: Costa, M., Goldberger, A. L., & Peng, C. K. (2002). Multiscale entropy
    analysis of complex physiologic time series. Physical review letters, 89(6), 068102.
[^Costa2015]: Costa, M. D., & Goldberger, A. L. (2015). Generalized multiscale entropy
    analysis: Application to quantifying the complex volatility of human heartbeat time
    series. Entropy, 17(3), 1197-1203.
"""
Base.@kwdef struct Regular <: MultiScaleAlgorithm
    f::Function = Statistics.mean
end

"""
    Composite <: MultiScaleAlgorithm
    Composite(; f::Function = Statistics.mean)

Composite multi-scale algorithm for multiscale entropy analysis (Wu et al.,
2013)[^Wu2013], used, with [`multiscale`](@ref) to compute, for example, composite
multiscale entropy (CMSE).

## Description

Given a scalar-valued input time series `x`, the composite multiscale algorithm,
like [`Regular`](@ref), downsamples and coarse-grains `x` by splitting it into
non-overlapping windows of length `s`, and then constructing downsampled time series by
applying the function `f` to each of the resulting length-`s` windows.

However, Wu et al. (2013)[^Wu2013] realized that for each scale `s`, there are actually `s`
different ways of selecting windows, depending on where indexing starts/ends.
These `s` different downsampled time series `D_t(s, f)` at each scale `s` are
constructed as follows:

```math
\\{ D_{k}(s) \\} = \\{ D_{t, k}(s) \\}_{t = 1}^{L}, = \\{ f \\left( \\bf x_{t, k} \\right) \\} =
\\left\\{
    {f\\left( (x_i)_{i = (t - 1)s + k}^{ts + k - 1} \\right)}
\\right\\}_{t = 1}^{L},
```

where `L = floor((N - s + 1) / s)` and `1 ≤ k ≤ s`, such that ``D_{i, k}(s)`` is the `i`-th
element of the `k`-th downsampled time series at scale `s`.

Finally, compute ``\\dfrac{1}{s} \\sum_{k = 1}^s g(D_{k}(s))``, where `g` is some summary
function, for example [`entropy`](@ref).

!!! note "Relation to [`Regular`](@ref)"
    The downsampled time series ``D_{t, 1}(s)`` constructed using the composite
    multiscale method is equivalent to the downsampled time series `D_{t}(s)` constructed
    using the [`Regular`](@ref) method, for which `k == 1` is fixed, such that only
    a single time series is returned.

[^Wu2013]: Wu, S. D., Wu, C. W., Lin, S. G., Wang, C. C., & Lee, K. Y. (2013). Time series
    analysis using composite multiscale entropy. Entropy, 15(3), 1069-1084.
"""
Base.@kwdef struct Composite <: MultiScaleAlgorithm
    f::Function = Statistics.mean
end

"""
    downsample(algorithm::MultiScaleAlgorithm, x::AbstractVector{T}, s::Int, args...;
        kwargs...)

Downsample and coarse-grain `x` to scale `s` according to the given multiscale `algorithm`.

Positional arguments `args` and keyword arguments `kwargs` are propagated to relevant
functions in `algorithm`, if applicable.

`The return type depends on `algorithm`. For example:

- [`Regular`](@ref) yields a single `Vector` per scale.
- [`Composite`](@ref) yields a `Vector{Vector}` per scale.
"""
function downsample(method::MultiScaleAlgorithm, x::AbstractVector{T}, s::Int, args...; kwargs...) where T
    f = method.f

    if s == 1
        return x
    else
        N = length(x)
        L = floor(Int, N / s)
        ys = zeros(T, L)

        for t = 1:L
            inds = ((t - 1)*s + 1):(t * s)
            ys[t] = @views f(x[inds], args...; kwargs...)
        end
        return ys
    end
end

function downsample(method::Composite, x::AbstractVector{T}, s::Int, args...; kwargs...) where T
    f = method.f

    if s == 1
        return [x]
    else
        N = length(x)
        # note: there must be a typo or error in Wu et al. (2013), because if we use
        # floor(N / s), as indicated in their paper, we'll get out of bounds errors.
        # The number of samples needs to take into consideratio how many unique ways
        # there are of selecting non-overlapping windows of length `s`. Hence,
        # we use floor((N - s + 1) / s) instead.
        L = floor(Int, (N - s + 1) / s)
        ys = [zeros(T, L) for i = 1:s]
        for k = 1:s
            for t = 1:L
                inds = ((t - 1)*s + k):(t * s + k - 1)
                ys[k][t] = @views f(x[inds], args...; kwargs...)
            end
        end
        return ys
    end
end

"""
    multiscale(e::Entropy, alg::MultiScaleAlgorithm, x::AbstractVector, est::ProbabilitiesEstimator;
        scalemax::Int = 8, normalize = false)

Compute the multi-scale entropy (Costa et al., 2002)[^Costa2002] of type `e` of
timeseries `x` using the given probabilities estimator `est`

Utilizes [`downsample`](@ref) to compute the entropy of coarse-grained, downsampled
versions of `x` for scale factors `1:maxscale`. The length of the most severely
downsampled version of `x` is `N ÷ maxscale`, while for scale factor `1`, the original
time series is considered.

If `normalize == true`, then compute normalized entropy (if that is possible for this
particular combination of entropy type and probability estimator).

[^Costa2002]: Costa, M., Goldberger, A. L., & Peng, C. K. (2002). Multiscale entropy
    analysis of complex physiologic time series. Physical review letters, 89(6), 068102.
"""
function multiscale end

function multiscale(e::Entropy, alg::Regular, x::AbstractVector, est::ProbabilitiesEstimator;
        maxscale::Int = 8, normalize = false)

    downscaled_timeseries = [downsample(alg, x, s) for s in 1:maxscale]
    hs = zeros(Float64, maxscale)
    if normalize
        hs = entropy_normalized.(Ref(e), downscaled_timeseries, Ref(est))
    else
        hs .= entropy.(Ref(e), downscaled_timeseries, Ref(est))
    end

    return hs
end

function multiscale(e::Entropy, alg::Composite, x::AbstractVector, est::ProbabilitiesEstimator;
    maxscale::Int = 10, normalize = false)

    downscaled_timeseries = [downsample(alg, x, s) for s in 1:maxscale]
    hs = zeros(Float64, maxscale)
    for s in 1:maxscale
        if normalize
            hs[s] = mean(entropy_normalized.(Ref(e), downscaled_timeseries[s], Ref(est)))
        else
            hs[s] = mean(entropy.(Ref(e), downscaled_timeseries[s], Ref(est)))
        end
    end

    return hs
end
