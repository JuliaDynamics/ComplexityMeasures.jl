"""
    DisequilibriumGenerator(probability_generator, x, init, rng)

A generator for repeated computation of [`disequilibrium`](@ref).

# Fields

- A probability generator `probability_generator`(see [`ProbabilityGenerator`](@ref)).
- Input data `x`.
- Initialization values `init`, given as a named tuple.
- A random number generator `rng` for reproducibility, which is applied if relevant.
"""
struct DisequilibriumGenerator{M, PG, X, A, R <: AbstractRNG}
    method::M
    probability_generator::PG
    x::X      # input data
    init::A   # pre-initialized things that speed up estimation
    rng::R    # random number generator object
end

"""
    disequilibriumgenerator(x, method, rng = Random.default_rng())

Generate a [`DisequilibriumGenerator`](@ref) for `x` using the given estimation `method`.
"""
function disequilibriumgenerator(x, method, rng = Random.default_rng())
    pg = probabilitygenerator(x, method, rng)
    init = ()
    return DisequilibriumGenerator(method, pg, x, init, rng)
end
