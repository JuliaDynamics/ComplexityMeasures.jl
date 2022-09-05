using Random
export EntropyGenerator
export entropygenerator


struct EntropyGenerator{S <: EntropyEstimator, X, A, R <: AbstractRNG}
    method::S # method with its input parameters
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
