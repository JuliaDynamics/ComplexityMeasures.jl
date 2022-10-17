using Distances

export statistical_complexity


"""
    statistical_complexity(x, est, distance = JSDivergence, normalize = true)


"""
function statistical_complexity(x::AbstractVector{T}, est::ProbabilitiesEstimator;
                                distance::AbstractFunction = JSDivergence,
                                normalize = true)

    # get probabilities from data
    probs = probabilities(x, est)

    # generate uniform distribution with N = length(p) bins
    uniform = ones(length(p)) * 1 / length(p)

    # calculate distance between p and uniform distribution
    dist = distance(probs, uniform)

    if normalize
        # generate "deterministic" distribution (only one completely full bin)
        determ = zeros(length(p))
        determ[1] = 1.0
        # distance to this distribution is the maximum possible distance to a uniform distribution
        return dist / distance(determ, uniform) * 
    end

    return dist
end


    
