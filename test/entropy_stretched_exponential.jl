# Analytical test cases from Anteneodo & Plastino (1999)
# -----------------------------------------------------

# Minimum value 0 occurs when there is complete certainty (exactly one outcome with
#  probability 1)
p = Probabilities([1.0])
@test entropy_streched_exponential(p) ≈ 0.0

# Maximal value occurs for uniform distribution
N = 10
p = Probabilities(repeat([1/N], N))
b = 2
η = 2.0
@test entropy_streched_exponential(p; η = η, base = b) ≈
    maxentropy_stretched_exponential(N, η = η, base = b)
