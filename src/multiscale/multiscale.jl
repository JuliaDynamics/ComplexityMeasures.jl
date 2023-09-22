function multiscale(alg::MultiScaleAlgorithm, e::ComplexityEstimator, x::AbstractVector;
        maxscale::Int = 8)

    downscaled_timeseries = [downsample(alg, s, x) for s in 1:maxscale]
    return complexity.(Ref(e), downscaled_timeseries)
end

function multiscale_normalized(alg::Regular, e::ComplexityEstimator, x::AbstractVector;
        maxscale::Int = 8)
    downscaled_timeseries = [downsample(alg, s, x) for s in 1:maxscale]
    return complexity_normalized.(Ref(e), downscaled_timeseries)
end

function multiscale(alg::Composite, e::ComplexityEstimator, x::AbstractVector;
        maxscale::Int = 8)

    downscaled_timeseries = [downsample(alg, s, x) for s in 1:maxscale]
    complexities = zeros(<:AbstractFloat, maxscale)
    for s in 1:maxscale
        complexities[s] = mean(complexity.(Ref(e), downscaled_timeseries[s]))
    end
    return complexities
end

function multiscale_normalized(alg::Composite, e::ComplexityEstimator, x::AbstractVector;
        maxscale::Int = 8)
    downscaled_timeseries = [downsample(alg, s, x) for s in 1:maxscale]
    complexities = zeros(<:AbstractFloat, maxscale)
    for s in 1:maxscale
        complexities[s] = mean(complexity_normalized.(Ref(e), downscaled_timeseries[s]))
    end
    return complexities
end
