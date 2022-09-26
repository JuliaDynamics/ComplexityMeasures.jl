export entropy_normalized

"""
    entropy_normalized(f::Function, x, est::ProbabilitiesEstimator, args...; kwargs...)

Convenience syntax for normalizing to the entropy of uniform probability distribution.
First estimates probabilities as `p::Probabilities = f(x, est, args...; kwargs...`),
then calls `entropy_normalized(f, p, args...; kwargs...)`.

Normalization is only defined for estimators for which [`alphabet_length`](@ref) is defined,
meaning that the total number of states or symbols is known beforehand.

    entropy_normalized(f::Function, p::Probabilities, est::ProbabilitiesEstimator, args...;
        kwargs...)

Normalize the entropy, as returned by the entropy function `f` called with the given
arguments (i.e. `f(p, args...; kwargs...)`), to the entropy of a uniform distribution,
inferring [`alphabet_length`](@ref) from `est`.

    entropy_normalized(f::Function, p::Probabilities, args...; kwargs...)

The same as above, but infers alphabet length from counting how many elements
are in `p` (zero probabilities are counted).

## Examples

Computing normalized entropy from scratch:

```julia
x = rand(100)
entropy_normalized(entropy_renyi, x, Dispersion())
```

Computing normalized entropy from pre-computed probabilities with known parameters:

```julia
x = rand(100)
est = Dispersion(m = 3, symbolization = GaussianSymbolization(c = 4))
p = probabilities(x, est)
entropy_normalized(entropy_renyi, p, est)
```

Computing normalized entropy, assumming there are `N = 10` total states:

```julia
N = 10
p = Probabilities(rand(10))
entropy_normalized(entropy_renyi, p, est)
```

!!! note "Normalized output range"
    For Rényi entropy (e.g. Kumar et al., 1986), and for Tsallis entropy (Tsallis, 1998),
    normalizing to the uniform distribution ensures that the entropy lies in
    the interval `[0, 1]`. For other entropies and parameter choices, the resulting entropy
    is not guaranteed to lie in `[0, 1]`. It is up to the user to decide whether
    normalizing to a uniform distribution makes sense for their use case.


[^Kumar1986]: Kumar, U., Kumar, V., & Kapur, J. N. (1986). Normalized measures of entropy.
    International Journal Of General System, 12(1), 55-69.
[^Tsallis1998]: Tsallis, C. (1988). Possible generalization of Boltzmann-Gibbs statistics.
    Journal of statistical physics, 52(1), 479-487.
"""
function entropy_normalized(f::Function, x, est::ProbEst, args...; kwargs...)
    N = alphabet_length(est)
    p_uniform = Probabilities(repeat([1/N], N))
    return f(x, est, args...; kwargs...) / f(p_uniform, args...; kwargs...)
end

function entropy_normalized(f::Function, x::Probabilities, est::ProbEst, args...; kwargs...)
    N = alphabet_length(est)
    p_uniform = Probabilities(repeat([1/N], N))
    return f(x, args...; kwargs...) / f(p_uniform; kwargs...)
end

function entropy_normalized(f::Function, x::Probabilities, args...; kwargs...)
    N = length(x)
    p_uniform = Probabilities(repeat([1/N], N))
    return f(x, args...; kwargs...) / f(p_uniform; kwargs...)
end
