using Random
rng = MersenneTwister(1234)

x = rand(1000)
xp = Probabilities(x)
@test_throws MethodError entropy(Shannon(), x) isa Real
@test entropy(Shannon(base = 10), xp) isa Real
@test entropy(Shannon(base = 2), xp) isa Real
@test entropy_maximum(Shannon(), 2) == 1

# Analytical tests
# -----------------------------------
# Minimal and equal to zero when probability distribution has only one element...
@test entropy(Shannon(), Probabilities([1.0])) ≈ 0.0

# or minimal when only one probability is nonzero and equal to 1.0
@test entropy(Shannon(), Probabilities([1.0, 0.0, 0.0, 0.0])) ≈ 0.0

# -----------------
# Direct estimators
# -----------------
# We just check if the estimator converge to true values for some distributions with
# analytically derivable entropy.
# Entropy to log with base b of a uniform distribution on [0, 1] = ln(1 - 0)/(ln(b)) = 0
U = 0.00
# Entropy with natural log of 𝒩(0, 1) is 0.5*ln(2π) + 0.5.
N = round(0.5*log(2π) + 0.5, digits = 2)

ev = Vasicek(m = 100, base = 2)
ee = Ebrahimi(m = 100, base = 2)
ev_n = Vasicek(m = 100, base = MathConstants.e)
ee_n = Ebrahimi(m = 100, base = MathConstants.e)

@test round(entropy(ev, rand(rng, 1000000)), digits = 2) == U
@test round(entropy(ev_n, randn(rng, 1000000)), digits = 2) == N
@test round(entropy(ee, rand(rng, 1000000)), digits = 2) == U
@test round(entropy(ee_n, randn(rng, 1000000)), digits = 2) == N

ea = AlizadehArghami(m = 100, base = MathConstants.e)
@test round(entropy(ea, rand(rng, 1000000)), digits = 2) == U
@test round(entropy(ea, randn(rng, 1000000)), digits = 2) == N
