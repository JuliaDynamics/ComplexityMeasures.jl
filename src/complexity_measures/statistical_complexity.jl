using Distances

export StatisticalComplexity, maximum_complexity_entropy, minimum_complexity_entropy

evaluate(distance::SemiMetric, a::Probabilities, b::Probabilities) = Distances.evaluate(distance, a.p, b.p)

"""
    StatisticalComplexity <: ComplexityMeasure
    StatisticalComplexity(est = SymbolicPermutation(),
                          distance = JSDivergence(),
                          entropy = Renyi())

Calculate the statistical complexity of a time series `x` as defined by Rosso et al. [^Rosso2007], based on 
permutation patterns.

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

For an ordinal distribution with a given permutation entropy, there is a minimum and maximum value that the
statistical complexity can take. The minimum and maximum statistical complexity as a function of the entropy
is calculated using [`minimum_complexity_entropy`](@ref) and [`maximum_complexity_entropy`](@ref).

Standard usage:
```julia
m, τ = 3, 1
est = SymbolicPermutation(; m, τ)
c = StatisticalComplexity(; est, entropy=Renyi())
x = randn(100)
statistical_complexity = complexity(c, x)
```

NOTE: I'm not sure if this API is congruent with the general complexity API, but I think
that, since the "standard" signature is also possible, it should be fine.
I have implemented the other signature because I find it redundant to calculate the entropy
twice, which is needed for the complexity anyway and usually calculated in combination with it.

Because the distributions of permutation symbols are needed both for the permutation entropy and the
statistical complexity, the same can also be achieved using
```julia
p = probabilities(est, x)
permutation_entropy = entropy(est, x)
statistical_complexity = complexity(c, p; h=permutation_entropy)
```

[^Rosso2007] Rosso, O. A., Larrondo, H. A., Martin, M. T., Plastino, A., & Fuentes, M. A. (2007). 
            [Distinguishing noise from chaos](https://doi.org/10.1103/PhysRevLett.99.154102).
            Physical review letters, 99(15), 154102. 
"""
Base.@kwdef struct StatisticalComplexity{P<:ProbabilitiesEstimator, M, E <: Entropy} <: ComplexityMeasure
    est::P= SymbolicPermutation()
    distance::M = JSDivergence()
    entropy::E = Renyi()
end

alphabet_length(c::StatisticalComplexity) = alphabet_length(c.est)

function complexity(c::StatisticalComplexity, p::Probabilities; h::Float64) 
    L = alphabet_length(c)
    # fill with the probability distribution zeros for empty bins, 
    # we need this to compare to a uniform distribution of m! patterns
    p = Probabilities(cat(zeros(L-length(p)), p.p, dims=1))
    # normalization is done automatically by Probabilities
    uniform = Probabilities(ones(L))

    dist = evaluate(c.distance, p, uniform)
    dist *= h

    return dist
end

function complexity(c::StatisticalComplexity, x::AbstractArray)
    p = probabilities(x, c.est)
    L = alphabet_length(c)
    h = entropy(c.entropy, p) / log(c.entropy.base, L)
    complexity(c, p; h)
end

function complexity_normalized(c::StatisticalComplexity, p::Probabilities; h::Float64)
    comp = complexity(c, p; h)
    L = alphabet_length(c)
    uniform = Probabilities(ones(L))
    # generate "deterministic" distribution (only one completely full bin)
    determ = Probabilities(Base.append!(zeros(L-1), 1.0))
    # distance to this distribution is the maximum possible distance to a uniform distribution
    return comp / evaluate(c.distance, determ, uniform)
end

function complexity_normalized(c::StatisticalComplexity, x::AbstractArray)
    comp = complexity(c, x)
    L = alphabet_length(c)
    uniform = Probabilities(ones(L))
    # generate "deterministic" distribution (only one completely full bin)
    determ = Probabilities(Base.append!(zeros(L-1), 1.0))
    # distance to this distribution is the maximum possible distance to a uniform distribution
    return comp / evaluate(c.distance, determ, uniform)
end

linearpermissiverange(start; stop, length) = length==1 ? (start:start) : range(start, stop=stop, length=length)

"""
    maximum_complexity_entropy(c::StatisticalComplexity; num=1)

Calculate the maximum complexity-entropy curve for the statistical complexity according to [^Rosso2007]
for `num * m!` different values of the normalized permutation entropy.

## Description
The way the statistical complexity is designed, there is a minimum and maximum possible complexity
for data with a given permutation entropy.
The calculation time of the maximum complexity curve grows as O((m!)^2), and thus takes very long for higher `m`.

This function is adapted from S. Sippels implementation in statcomp [^statcomp].

[^Rosso2007] Rosso, O. A., Larrondo, H. A., Martin, M. T., Plastino, A., & Fuentes, M. A. (2007). 
            [Distinguishing noise from chaos](https://doi.org/10.1103/PhysRevLett.99.154102).
            Physical review letters, 99(15), 154102.

[^statcomp] Sippel, S., Lange, H., Gans, F. (2019).
            [statcomp: Statistical Complexity and Information Measures for Time Series Analysis](https://cran.r-project.org/web/packages/statcomp/index.html)
"""
function maximum_complexity_entropy(c::StatisticalComplexity; num::Int=1)
    
    L = alphabet_length(c)
    # in these we'll write the entropy (h) and corresponding max. complexity (c) values
    hs, cs = zeros(L-1, num), zeros(L-1, num)
    norm = log(c.entropy.base, L)
    for i in 1:L-1
        p = zeros(L)
        prob_params = linearpermissiverange(0; stop=1/L, length=num)
        for k in 1:num
            p[1] = prob_params[k]
            for j in 1:L-i
                p[j] = (1-prob_params[k]) / (L-i)
            end
            p_k = Probabilities(p)
            h = entropy(c.entropy, p_k) / norm
            hs[i, k] = h
            cs[i, k] = complexity_normalized(c, p_k; h)
        end
    end
    hs = vcat(hs...)
    cs = vcat(cs...)
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

    L = alphabet_length(c)
    prob_params = linearpermissiverange(1/L; stop=1, length=num)
    hs = Float64[]
    cs = Float64[]
    norm = log(c.entropy.base, L)

    for i in 1:num
        p_i = ones(L) * (1-prob_params[i]) / (L-1)
        p_i[1] = prob_params[i]
        p_i = Probabilities(p_i)
        h = entropy(c.entropy, p_i) / norm
        push!(hs, h)
        push!(cs, complexity_normalized(c, p_i; h))
    end
    return reverse(hs), reverse(cs)
end