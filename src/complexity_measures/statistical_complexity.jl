using Distances

export statistical_complexity, maximum_complexity_entropy, minimum_complexity_entropy

evaluate(distance::SemiMetric, a::Probabilities, b::Probabilities) = evaluate(distance, vec(a), vec(b))

"""
    statistical_complexity(x, est; kwargs...) -> C_JS

Calculate the statistical complexity of a time series `x` as defined by Rosso et al. [^Rosso2007], based on 
permutation patterns.

## Keyword Arguments
* `distance = JSDivergence()` : distance measure to calculate distance of ordinal distribution to uniform distribution
* `normalize = true` : whether to normalize the JS divergence with the maximum distance to a uniform distribution
* `entropy = Renyi()` : entropy to use for the calculation of the permutation entropy

## Description
The statistical complexity ``C_{JS}[P]`` is defined based on the distribution ``P`` of 
ordinal patterns
```math
C_{JS}[P] = Q_J[P, P_e] H_S[P]
```
where ``Q_J`` is originally the normalized Jensen-Shannon divergence of the ordinal distribution ``P`` to 
a uniform distribution ``P_e`` with the same numer of bins, and ``H_S`` is the normalized permutation entropy, 
`entropy_normalized(x, SymbolicPermutation())`.
In this implementation, the distance measure defaults to the Jensen-Shannon divergence, but other (semi-)metrics can also be
chosen. The type of entropy is also a free parameter, but defaults to the (originally used) Renyi entropy.


[^Rosso2007] Rosso, O. A., Larrondo, H. A., Martin, M. T., Plastino, A., & Fuentes, M. A. (2007). 
            [Distinguishing noise from chaos](https://doi.org/10.1103/PhysRevLett.99.154102).
            Physical review letters, 99(15), 154102. 
"""
function statistical_complexity(p::Probabilities, est::ProbabilitiesEstimator, h::Float64;
                                distance::SemiMetric = JSDivergence(),
                                normalize::Bool = true) 

    L = factorial(est.m)
    # fill with the probability distribution zeros for empty bins, 
    # we need this to compare to a uniform distribution of m! patterns
    # we put p at the end because it might be just a float, and append
    # is not defined for this order of arguments
    append!(zeros(L-length(p)), p.p)
    # normalization is done automatically by Probabilities
    uniform = Probabilities(ones(L))

    dist = evaluate(distance, p, uniform)
    dist *= h

    if normalize
        # generate "deterministic" distribution (only one completely full bin)
        determ = Probabilities(append!(zeros(L-1), 1.0))
        # distance to this distribution is the maximum possible distance to a uniform distribution
        return dist / evaluate(distance, determ, uniform)
    end

    return dist
end

function statistical_complexity(p::Probabilities, est::ProbabilitiesEstimator;
                                entropy_type::Entropy=Renyi(), kwargs...)
    L = factorial(est.m)
    h = entropy(entropy_type, p) / log(entropy_type.base, L)
    return statistical_complexity(p, est, h; kwargs...)
end

statistical_complexity(x::AbstractArray, est::ProbabilitiesEstimator; kwargs...) = statistical_complexity(probabilities(x, est), est; kwargs...)

linearpermissiverange(start; stop, length) = length==1 ? (start:start) : range(start, stop=stop, length=length)

"""
    maximum_complexity_entropy(est::ProbabilitiesEstimator; num=1, entropy_type=Renyi())

Calculate the maximum complexity-entropy curve for 
"""
function maximum_complexity_entropy(est::ProbabilitiesEstimator; num::Int=1, entropy_type::Entropy=Renyi())
    
    L = alphabet_length(est)
    # in these we'll write the entropy (h) and corresponding max. complexity (c) values
    hs, cs = zeros(L-1, num), zeros(L-1, num)
    norm = log(entropy_type.base, L)
    for i in 1:L-1
        p = zeros(L)
        prob_params = linearpermissiverange(0; stop=1/L, length=num)

        for k in 1:num
            p[1] = prob_params[k]
            for j in 1:L-i
                p[j] = (1-prob_params[k]) / (L-i)
            end
            p_k = Probabilities(p)
            h = entropy(entropy_type, p_k) / norm
            hs[i, k] = h
            cs[i, k] = statistical_complexity(p_k, est, h)
        end
    end
    hs = vcat(hs...)
    cs = vcat(cs...)
    args = sortperm(hs)
    return hs[args], cs[args]
end

function minimum_complexity_entropy(est::ProbabilitiesEstimator; num::Int=100, entropy_type::Entropy=Renyi())

    L = alphabet_length(est)
    prob_params = linearpermissiverange(1/L; stop=1, length=num)
    hs = Float64[]
    cs = Float64[]
    norm = log(entropy_type.base, L)

    for i in 1:num
        p_i = ones(L) * (1-prob_params[i]) / (L-1)
        p_i[1] = prob_params[i]
        p_i = Probabilities(p_i)
        h = entropy(entropy_type, p_i) / norm
        push!(hs, h)
        push!(cs, statistical_complexity(p_i, est, h))
    end
    return hs, cs
end