export CompositeDownsampling

"""
    CompositeDownsampling <: MultiScaleAlgorithm
    CompositeDownsampling(; f::Function = Statistics.mean)

Composite multi-scale algorithm for multiscale entropy analysis [Wu2013](@cite), used,
with [`multiscale`](@ref) to compute, for example, composite multiscale entropy (CMSE).

## Description

Given a scalar-valued input time series `x`, the composite multiscale algorithm,
like [`RegularDownsampling`](@ref), downsamples and coarse-grains `x` by splitting it into
non-overlapping windows of length `s`, and then constructing downsampled time series by
applying the function `f` to each of the resulting length-`s` windows.

However, [Wu2013](@citet) realized that for each scale `s`, there are actually `s`
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
function, for example [`information`](@ref) or [`complexity`](@ref).

!!! note "Relation to RegularDownsampling"
    The downsampled time series ``D_{t, 1}(s)`` constructed using the composite
    multiscale method is equivalent to the downsampled time series ``D_{t}(s)`` constructed
    using the [`RegularDownsampling`](@ref) method, for which `k == 1` is fixed, such that only
    a single time series is returned.

See also: [`RegularDownsampling`](@ref).
"""
Base.@kwdef struct CompositeDownsampling <: MultiScaleAlgorithm
    f::Function = Statistics.mean
end

function downsample(method::CompositeDownsampling, s::Int, x::AbstractVector{T}, args...;
        kwargs...) where T
    verify_scale_level(method, s, x)

    f = method.f
    ET = eltype(one(1.0)) # always return floats, even if input is e.g. integer-valued

    if s == 1
        return [ET.(x)]
    else
        N = length(x)
        # Because input time series are finite, there is always a minimum number of windows
        # that we can construct at a given scale. We restrict the number of windows
        # considered at each scale to this minimum to ensure windows are well-defined,
        # i.e. we're not trying to summarize data at indices outside the input data,
        # which would give out-of-bounds errors.
        #
        # For example, if the input is [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], then we sample
        # the following subvectors at different scales:
        # Scale 3:
        #    start index 1: [1, 2, 3], [4, 5, 6], [7, 8, 9]
        #    start index 2: [2, 3, 4], [5, 6, 7], [8, 9, 10]
        #    start index 3: [3, 4, 5], [6, 7, 8]
        #    Only two windows are possible for start index 3, so `min_possible_windows = 2`
        # Scale 4:
        #    start index 1: [1, 2, 3, 4], [5, 6, 7, 8]
        #    start index 2: [2, 3, 4, 5], [6, 7, 8, 9]
        #    start index 3: [3, 4, 5, 6], [7, 8, 9, 10]
        #    start index 4: [4, 5, 6, 7]
        #    Only one window is possible for start index 4, so `min_possible_windows = 1`
        min_possible_windows = floor(Int, (N - s + 1) / s)

        ys = [zeros(ET, min_possible_windows) for i = 1:s]
        for k = 1:s
            for t = 1:min_possible_windows
                inds = ((t - 1)*s + k):(t * s + k - 1)
                ys[k][t] = @views f(x[inds], args...; kwargs...)
            end
        end
        return ys
    end
end

function apply_multiscale(alg::CompositeDownsampling, f::Function, args...;
        maxscale = 8)
    # Assume last argument is the input data.
    downscaled_timeseries = [downsample(alg, s, last(args)) for s in 1:maxscale]

    # Use all args for estimation, except the last argument, which is the input data.
    estimation_args = @views args[1:end-1]
    hs = zeros(Float64, maxscale)
    for s in 1:maxscale
        x = downscaled_timeseries[s] # contains an array of time series
        hs[s] = mean([f(estimation_args..., tsᵢ) for tsᵢ in x])
    end

    return hs
end
