using Distances
using ComplexityMeasures: ProbabilitiesEstimator

export StatisticalComplexity, minimum_complexity_entropy, maximum_complexity_entropy

"""
    StatisticalComplexity <: ComplexityEstimator
    StatisticalComplexity([x]; kwargs...)

An estimator for the statistical complexity and entropy according to [^Rosso2007], used with [`complexity`](@ref).

## Keyword arguments
- `est::ProbabilitiesEstimator = SymbolicPermutation()`: which estimator to use to get the probabilities
- `dist<:SemiMetric = JSDivergence()`: the distance measure between the estimated probability
    distribution and a uniform distribution with the same maximal number of bins

## Description
Statistical Complexity is defined as 
```math
C_q[P] = \\mathcal{H}_q\\cdot \\mathcal{Q}_q[P],
```
where `Q_q` is a "disequilibrium" obtained from a distance-measure and
`H_q` a disorder measure.
In the original paper[^Rosso2007], this complexity measure was defined
via an ordinal pattern-based probability distribution, the Shannon entropy 
and the Jensen-Shannon divergence as a distance measure.
This implementation allows for a generalization of the
complexity measure as developed in [^Rosso2013].
Here, `H_q` can be the (q-order) Shannon-, Renyi or Tsallis
entropy and Q_q based either on the Euclidean, Wooters, Kullback,
q-Kullback, Jensen or q-Jensen distance.

[^Rosso2007]: Rosso, O. A. et al. (2007). Distinguishing Noise from Chaos. 
    Physical Review Letters 99, no. 15: 154102. https://doi.org/10.1103/PhysRevLett.99.154102.
[^Rosso2013]: Rosso, O. A. (2013) Generalized Statistical Complexity: A New Tool for Dynamical Systems.

"""
Base.@kwdef struct StatisticalComplexity{E, D, H} <: ComplexityEstimator
    dist::D = JSDivergence()
    est::E = SymbolicPermutation()
    entr::H = Renyi()

    function StatisticalComplexity(dist::D, est::E, entr::H) where {E, D, H}
        new{E, D, H}(dist, est, entr)
    end
    function StatisticalComplexity(x::AbstractVector;
            dist::SemiMetric, est::ProbabilitiesEstimator,
            entr::EntropyDefinition)
        StatisticalComplexity(dist, est, entr)
    end
end

function complexity(c::StatisticalComplexity, x::AbstractVector{T}) where T
    (; dist, est, entr) = c

    p = probabilities(est, x)

    H_q = entropy_normalized(entr, est, x)

    L = total_outcomes(est, x)

    # calculate distance between calculated distribution and uniform one
    D_q = evaluate(dist, p.p, fill(1.0/L, size(p)))

    # generate distribution with just one filled bin
    deterministic = zeros(size(p))
    deterministic[1] = 1

    D_max = evaluate(dist, deterministic, fill(1.0/L, size(p)))
    C_q = D_q / D_max * H_q

    return C_q
end

function complexity(c::StatisticalComplexity, p::Probabilities)
    (; dist, est, entr) = c
    if !(est isa SymbolicPermutation || est isa SymbolicWeightedPermutation || est isa SymbolicAmplitudeAwarePermutation)
        throw(
            ArgumentError(
                "To estimate the statistical complexity directly from a probabiliy distribution, "*
                "the probabilities estimator must be one of"*
                "`SymbolicPermutation`, `SymbolicAmplitudeAwarePermutation` or `SymbolicWeightedPermutation`"
            )
        )
    end
    L = total_outcomes(est, randn(100))
    norm = log(entr.base, L)
    H_q = entropy(entr, p) / norm

    # calculate distance between calculated distribution and uniform one
    D_q = evaluate(dist, p.p, fill(1.0/L, size(p)))

    # generate distribution with just one filled bin
    deterministic = zeros(size(p))
    deterministic[1] = 1

    D_max = evaluate(dist, deterministic, fill(1.0/L, size(p)))
    C_q = D_q / D_max * H_q

    return C_q
end

function fill_probs_k!(p, prob_params, L, i, k)
    p.p[1] = prob_params[k] # why set first element here if overwriting it in the loop below?
    # if we know that p has sufficient length relative to L and i,
    # @inbounds can save some computation time by skipping bounds checking.
    @inbounds for j = 1:(L - i) 
        p.p[j] = (1 - prob_params[k]) / (L - i)
    end
end

linearpermissiverange(start; stop, length) = length==1 ? (start:start) : range(start, stop=stop, length=length)

"""
    maximum_complexity_entropy(c::StatisticalComplexity; num=1)
Calculate the maximum complexity-entropy curve for the statistical complexity according to [^Rosso2007]
for `num * total_outcomes(c.est)` different values of the normalized permutation entropy.

## Description
The way the statistical complexity is designed, there is a minimum and maximum possible complexity
for data with a given permutation entropy.
The calculation time of the maximum complexity curve grows as `O(total_outcomes(c.est)^2)`, and thus takes
very long for high numbers of outcomes.
This function is adapted from S. Sippels implementation in statcomp [^statcomp].

[^Rosso2007] Rosso, O. A., Larrondo, H. A., Martin, M. T., Plastino, A., & Fuentes, M. A. (2007).
            [Distinguishing noise from chaos](https://doi.org/10.1103/PhysRevLett.99.154102).
            Physical review letters, 99(15), 154102.
[^statcomp] Sippel, S., Lange, H., Gans, F. (2019).
            [statcomp: Statistical Complexity and Information Measures for Time Series Analysis](https://cran.r-project.org/web/packages/statcomp/index.html)
"""
function maximum_complexity_entropy(c::StatisticalComplexity; num::Int = 1)
    L = total_outcomes(c.est, randn(10))
    norm = log(c.entr.base, L)
    # avoid having to resize later by just making result containers vectors straight away.
    hs = zeros((L - 1) * num)
    cs = zeros((L - 1) * num)

    p = Probabilities(zeros(L), true) # can't normalize zeros, so let's pretend this is already normalized
    prob_params = linearpermissiverange(0; stop = 1 / L, length = num)

    j = 1
    for i in 1:(L - 1)
        p.p .= 0.0 # Note 0.0, not 0 (the elements in `p` are floats, so we should re-fill with floats to avoid conversions)
        for k in 1:num
            # Does this function ensure sum(p) == 1? If not, we need to normalize `p` afterwards, because `entropy` requires
            # normalized probabilities (i.e. summing to 1)
            fill_probs_k!(p, prob_params, L, i, k) 
            hs[j] = entropy(c.entr, p) / norm
            cs[j] = complexity(c, p)
            j += 1
        end
    end
    args = sortperm(hs)
    return hs[args], cs[args]
end

"""
    minimum_complexity_entropy(c::StatisticalComplexity; num=100) -> entropy, complexity
Calculate the maximum complexity-entropy curve for the statistical complexity according to [^Rosso2007]
for `num` different values of the normalized permutation entropy.

## Description
The way the statistical complexity is designed, there is a minimum and maximum possible complexity
for data with a given permutation entropy.
Here, the lower bound of the statistical complexity is calculated as a function of the permutation entropy
This function is adapted from S. Sippels implementation in statcomp [^statcomp].

[^Rosso2007] Rosso, O. A., Larrondo, H. A., Martin, M. T., Plastino, A., & Fuentes, M. A. (2007).
            [Distinguishing noise from chaos](https://doi.org/10.1103/PhysRevLett.99.154102).
            Physical review letters, 99(15), 154102.
[^statcomp] Sippel, S., Lange, H., Gans, F. (2019).
            [statcomp: Statistical Complexity and Information Measures for Time Series Analysis](https://cran.r-project.org/web/packages/statcomp/index.html)
"""
function minimum_complexity_entropy(c::StatisticalComplexity; num::Int=1000)

    L = total_outcomes(c.est, randn(10))
    prob_params = linearpermissiverange(1/L; stop=1, length=num)
    hs = Float64[]
    cs = Float64[]

    deterministic = zeros(L)
    deterministic[1] = 1
    d_max = evaluate(c.dist, deterministic, fill(1.0/L, L))
    h_max = entropy(c.entr, Probabilities(ones(L)/L))

    for i in 1:num
        p_i = ones(L) * (1-prob_params[i]) / (L-1)
        p_i[1] = prob_params[i]
        p_i = Probabilities(p_i)
        h = entropy(c.entr, p_i)
        push!(hs, h/h_max)
        d = evaluate(c.dist, p_i.p, fill(1.0/L, L))
        push!(cs, d / d_max * h / h_max)
    end
    return reverse(hs), reverse(cs)
end