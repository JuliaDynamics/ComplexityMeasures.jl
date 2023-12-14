using ComplexityMeasures
using Test
using Random
rng = Xoshiro(1234)

# Constructors
@test Probabilities(rand(rng, 10)) isa Probabilities
outs = collect(1:10)
@test Probabilities(rand(rng, 10), outs) isa Probabilities
@test Probabilities(rand(rng, 10), outs, :x1) isa Probabilities
@test Probabilities(rand(rng, 10), (outs,)) isa Probabilities
@test Probabilities(rand(rng, 10), (outs,), (:x1, )) isa Probabilities

# Enough data that all outcomes should be covered
x = rand(rng, 10000)
outcomemodel = OrdinalPatterns(m = 3)
est = RelativeAmount()
p = probabilities(est, outcomemodel, x)
Ω = outcome_space(est, outcomemodel, x)

# Different ways of obtaining the outcomes (the latter two intended for use
# in CausalityTools.jl with multi-input measures).
@test sort(outcomes(p)) == Ω
@test sort(outcomes(p, 1)) == Ω
@test outcomes(p, 1:1)[1] == Ω

# Error messages.
struct MyEstimator <: ProbabilitiesEstimator end
@test_throws MethodError probabilities(MyEstimator(), x)
