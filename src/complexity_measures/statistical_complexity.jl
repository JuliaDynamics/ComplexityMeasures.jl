using Distances

export statistical_complexity


"""
    statistical_complexity(x, est, distance = JSDivergence(), normalize = true, entropy_func=Renyi())

This function calculates the statistical complexity of a time series `x` as defined by Rosso et al. [^Rosso2007].
It calculates the statistical complexity based on permutation patterns. The statistical complexity ``C_{JS}[P]`` 
is defined based on the distribution ``P`` of ordinal patterns
```math
C_{JS}[P] = Q_J[P, P_e] H_S[P]
```
where ``Q_J`` is originally the normalized Jensen-Shannon divergence of the ordinal distribution to a uniform distribution,
and ``H_S`` is the normalized permutation entropy.

In this implementation, the distance measure defaults to the Jensen-Shannon divergence, but other (semi-)metrics can also be
chosen. The type of entropy defaults to the Renyi entropy.

If `normalize = true`, the complexity is divided by the maximum distance of the ordinal distribution to a uniform distribution,
otherwise this normalization step is skipped.

[^Rosso2007] Rosso et al. (2007). Distinguishing noise from chaos. https://doi.org/10.1103/PhysRevLett.99.154102
"""
function statistical_complexity(x::AbstractArray, est::ProbabilitiesEstimator;
                                distance::SemiMetric = JSDivergence(),
                                normalize::Bool = true,
                                entropy_func::Entropy = Renyi()) 

    # get probabilities from data
    probs = probabilities(x, est)
    # fill with zeros for empty bins, we need this to compare to a uniform distribution of m! patterns
    probs = cat(probs, zeros(factorial(est.m)-length(probs)), dims=1)
    # generate uniform distribution with N = length(p) bins
    uniform = ones(length(probs)) * 1 / length(probs)

    # calculate distance between p and uniform distribution
    dist = evaluate(distance, vec(probs), uniform)

    if normalize
        # generate "deterministic" distribution (only one completely full bin)
        determ = zeros(length(probs))
        determ[1] = 1.0
        # distance to this distribution is the maximum possible distance to a uniform distribution
        return dist / evaluate(distance, determ, uniform) * entropy_normalized(entropy_func, x, est)
    end

    return dist * entropy_normalized(entropy_func, x, est)
end
