using Test
using Random
rng = Xoshiro(1234)
# Enough data that all outcomes should be covered
x = rand(rng, 10000)
outcomemodel = OrdinalPatterns(m = 3)
est = RelativeAmount()
p = probabilities(est, outcomemodel, x)
Ω = outcome_space(est, outcomemodel, x)

# With the given `x` and `outcomemodel`, all outcomes should be covered
@test missing_outcomes(outcomemodel, x; all = true) == 0
@test missing_outcomes(outcomemodel, x; all = false) == 0
@test missing_outcomes(est, outcomemodel, x; all = true) == 0
@test missing_outcomes(est, outcomemodel, x; all = false) == 0
