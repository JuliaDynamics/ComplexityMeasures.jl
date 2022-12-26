function multiscale(alg::MultiScaleAlgorithm, e::ComplexityMeasure, x::AbstractVector;
        maxscale::Int = 8)

    downscaled_timeseries = [downsample(alg, s, x) for s in 1:maxscale]
    return complexity.(Ref(e), downscaled_timeseries)
end

function multiscale_normalized(alg::Regular, e::ComplexityMeasure, x::AbstractVector;
        maxscale::Int = 8)
    downscaled_timeseries = [downsample(alg, s, x) for s in 1:maxscale]
    return complexity_normalized.(Ref(e), downscaled_timeseries)
end

function multiscale(alg::Composite, e::ComplexityMeasure, x::AbstractVector;
        maxscale::Int = 8)

    downscaled_timeseries = [downsample(alg, s, x) for s in 1:maxscale]
    complexities = zeros(Float64, maxscale)
    for s in 1:maxscale
        complexities[s] = mean(complexity.(Ref(e), downscaled_timeseries[s]))
    end
    return complexities
end

function multiscale_normalized(alg::Composite, e::ComplexityMeasure, x::AbstractVector;
        maxscale::Int = 8)
    downscaled_timeseries = [downsample(alg, s, x) for s in 1:maxscale]
    complexities = zeros(Float64, maxscale)
    for s in 1:maxscale
        complexities[s] = mean(complexity_normalized.(Ref(e), downscaled_timeseries[s]))
    end
    return complexities
end
