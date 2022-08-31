using Random
export EntropyGenerator, ProbabilityGenerator
export entropygenerator, probabilitygenerator

struct ProbabilityGenerator{S <: Union{ProbabilitiesEstimator}, X, A, R <: AbstractRNG}
    method::S # method with its input parameters
    x::X      # input data
    init::A   # pre-initialized things that speed up probability estimation
    rng::R    # random number generator object
end

struct EntropyGenerator{S, PG, X, A, R <: AbstractRNG}

    method::S # method with its input parameters
        # A probability estimator can be supplied if the probability estimation is separate
    # from the entropy computation step. This may not be the case for all methods, so
    # it defaults to nothing.
    probability_generator::PG

    x::X      # input data
    init::A   # pre-initialized things that speed up entropy estimation
    rng::R    # random number generator object

end


"""
    entropygenerator(x, method::EntropyEstimator[, rng]) → sg::EntropyGenerator

Initialize a generator that computes entropies of `x` on demand, based on the given `method`.
This is efficient, because for most methods some things can be initialized and reused
for every computation. Optionally you can provide an `rng::AbstractRNG` object that will
control random number generation and hence establish reproducibility of the
generated entropy values, if they rely on random number generation. By default
`Random.default_rng()` is used.

Note: not all entropy estimators have this functionality enabled yet. The documentation
strings for individual methods indicate whether entropy generators are available.
To compute entropy using a generator, call `eg` as a function with the optional `base`
argument, e.g.

```julia
eg = entropygenerator(x, method)
h = eg(; base = 2)
```
"""
function entropygenerator end

"""
    probabilitiesgenerator(x, method, [, rng]) → sg::ProbabilitiesGenerator

The same as [`entropygenerator`](@ref), but initializes a `ProbabilitiesGenerator` that
can be used for repeated computation of probabilities.
"""
function probabilitygenerator end
