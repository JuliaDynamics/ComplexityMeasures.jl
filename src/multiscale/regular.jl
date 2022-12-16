export Regular

"""
    Regular <: MultiScaleAlgorithm
    Regular(; f::Function = Statistics.mean)

The original multi-scale algorithm for multiscale entropy analysis (Costa et al.,
2022)[^Costa2002], which yields a single downsampled time series per scale `s`.

## Description

Given a scalar-valued input time series `x`, the `Regular` multiscale algorithm downsamples
and coarse-grains `x` by splitting it into non-overlapping windows of length `s`, and
then constructing a new downsampled time series ``D_t(s, f)`` by applying the function `f`
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

- `f == Statistics.mean` yields the original first-moment multiscale sample entropy (Costa
    et al., 2002)[^Costa2002].
- `f == Statistics.var` yields the generalized multiscale sample entropy (Costa &
    Goldberger, 2015)[^Costa2015], which uses the second-moment (variance) instead of the
    mean.

See also: [`Composite`](@ref).

[^Costa2002]: Costa, M., Goldberger, A. L., & Peng, C. K. (2002). Multiscale entropy
    analysis of complex physiologic time series. Physical review letters, 89(6), 068102.
[^Costa2015]: Costa, M. D., & Goldberger, A. L. (2015). Generalized multiscale entropy
    analysis: Application to quantifying the complex volatility of human heartbeat time
    series. Entropy, 17(3), 1197-1203.
"""
Base.@kwdef struct Regular <: MultiScaleAlgorithm
    f::Function = Statistics.mean
end

function downsample(method::Regular, s::Int, x::AbstractVector{T}, args...; kwargs...) where T
    f = method.f
    verify_scale_level(method, s, x)

    ET = eltype(one(1.0)) # consistently return floats, even if input is e.g. integer-valued
    if s == 1
        return x
    else
        N = length(x)
        L = floor(Int, N / s)
        ys = zeros(ET, L)

        for t = 1:L
            inds = ((t - 1)*s + 1):(t * s)
            ys[t] = @views f(x[inds], args...; kwargs...)
        end
        return ys
    end
end

function multiscale(alg::Regular, e::Entropy,
        est::Union{ProbabilitiesEstimator, EntropyEstimator},
        x::AbstractVector;
        maxscale::Int = 8)

    downscaled_timeseries = [downsample(alg, s, x) for s in 1:maxscale]
    return entropy.(Ref(e), Ref(est), downscaled_timeseries)
end

function multiscale_normalized(alg::Regular, e::Entropy,
        est::ProbabilitiesEstimator, x::AbstractVector,;
        maxscale::Int = 8)

    downscaled_timeseries = [downsample(alg, s, x) for s in 1:maxscale]
    return entropy_normalized.(Ref(e), Ref(est), downscaled_timeseries)
end
