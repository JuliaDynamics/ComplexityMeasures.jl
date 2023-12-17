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
p_generic = probabilities(est, outcomemodel, x)
p_decoded, outs = probabilities_and_outcomes(est, outcomemodel, x)
Ω = outcome_space(est, outcomemodel, x)

# Different ways of obtaining the outcomes (the latter two intended for use
# in CausalityTools.jl with multi-input measures).
@test sort(outcomes(p_decoded)) == Ω
@test sort(outcomes(p_generic)) == Outcome(1):1:Outcome(length(Ω))

@test sort(outcomes(p_decoded, 1)) == Ω
@test outcomes(p_decoded, 1:1)[1] == Ω

# Error messages.
struct MyEstimator <: ProbabilitiesEstimator end
@test_throws MethodError probabilities(MyEstimator(), x)
