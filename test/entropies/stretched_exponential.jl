# Analytical test cases from Anteneodo & Plastino (1999)
# -----------------------------------------------------

# Minimum value 0 occurs when there is complete certainty (exactly one outcome with
#  probability 1)
p = Probabilities([1.0])
@test entropy(StretchedExponential(), p) ≈ 0.0

# Maximal value occurs for uniform distribution
N = 10
up = Probabilities(repeat([1/N], N))
b = 2
η = 2.0

@test entropy(StretchedExponential(η = η, base = b), up) ≈
    maximum(StretchedExponential(η = η, base = b), N)

# An experimental time series and probabilities estimator that gives a uniform
# probability distribution.
x = [repeat([0, 1], 5); 0]
est = SymbolicPermutation(m = 2)
@test entropy(StretchedExponential(η = η, base = b), x, est) ≈
    maximum(StretchedExponential(η = η, base = b), total_outcomes(est))
