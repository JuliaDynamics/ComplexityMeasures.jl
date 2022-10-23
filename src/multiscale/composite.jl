export Composite

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
function, for example [`entropy`](@ref) or [`complexity`](@ref).

!!! note "Relation to Regular"
    The downsampled time series ``D_{t, 1}(s)`` constructed using the composite
    multiscale method is equivalent to the downsampled time series ``D_{t}(s)`` constructed
    using the [`Regular`](@ref) method, for which `k == 1` is fixed, such that only
    a single time series is returned.

See also: [`Regular`](@ref).

[^Wu2013]: Wu, S. D., Wu, C. W., Lin, S. G., Wang, C. C., & Lee, K. Y. (2013). Time series
    analysis using composite multiscale entropy. Entropy, 15(3), 1069-1084.
"""
Base.@kwdef struct Composite <: MultiScaleAlgorithm
    f::Function = Statistics.mean
end

function downsample(method::Composite, x::AbstractVector{T}, s::Int, args...; kwargs...) where T
    f = method.f
    ET = eltype(one(1.0)) # consistently return floats, even if input is e.g. integer-valued

    if s == 1
        return ET.(x)
    else
        N = length(x)
        # note: there must be a typo or error in Wu et al. (2013), because if we use
        # floor(N / s), as indicated in their paper, we'll get out of bounds errors.
        # The number of samples needs to take into consideratio how many unique ways
        # there are of selecting non-overlapping windows of length `s`. Hence,
        # we use floor((N - s + 1) / s) instead.
        L = floor(Int, (N - s + 1) / s)
        ys = [zeros(ET, L) for i = 1:s]
        for k = 1:s
            for t = 1:L
                inds = ((t - 1)*s + k):(t * s + k - 1)
                ys[k][t] = @views f(x[inds], args...; kwargs...)
            end
        end
        return ys
    end
end

# TODO: make a separate multiscale_normalized?
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


function multiscale(e::ComplexityMeasure, alg::Composite, x::AbstractVector;
        maxscale::Int = 10, normalize = false)

    downscaled_timeseries = [downsample(alg, x, s) for s in 1:maxscale]
    complexities = zeros(Float64, maxscale)
    for s in 1:maxscale
        if normalize
            complexities[s] =
                mean(complexity_normalized.(Ref(e), downscaled_timeseries[s]))
        else
            complexities[s] = mean(complexity.(Ref(e), downscaled_timeseries[s]))
        end
    end

    return complexities
end
