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

# ----------------------------------------------------------------
# Base extensions 
# ----------------------------------------------------------------
p =  Probabilities(rand(rng, 10), outs)
# extend base Array interface:
@test length(p) == length(p.p)
@test size(p) == size(p.p)
@test eachindex(p) == eachindex(p.p)
@test eltype(p) == eltype(p.p)
@test parent(p) == parent(p.p)
@test firstindex(p) == firstindex(p.p)
@test lastindex(p) == lastindex(p.p)
@test vec(p) == vec(p.p)
@test getindex(c, 2) == getindex(c.cts, 2)
@test iterate(p) == iterate(p.p)
@test sort(p) == sort(p.p)

# The number of probabilities and outcomes must match.
@test_throws ArgumentError Probabilities(rand(1:3, 10), (1:9,))

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

# ----------------------------------------------------------------
# Pretty printing
# ----------------------------------------------------------------
p = Probabilities(rand(3))
out_capture = repr(MIME("text/plain"), p)
s = split(out_capture, '\n')
@test contains(first(s), "Probabilities{Float64,1} over 3 outcomes")
@test contains(s[2], "Outcome(1)")
@test contains(last(s), "Outcome(3)")