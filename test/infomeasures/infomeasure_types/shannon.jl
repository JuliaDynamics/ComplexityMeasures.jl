using Random
rng = MersenneTwister(1234)

x = rand(1000)
xp = Probabilities(x)
@test_throws MethodError information(Shannon(), x) isa Real
@test information(Shannon(base = 10), xp) isa Real
@test information(Shannon(base = 2), xp) isa Real
@test information_maximum(Shannon(), 2) == 1

# Analytical tests
# -----------------------------------
# Minimal and equal to zero when probability distribution has only one element...
@test information(Shannon(), Probabilities([1.0])) ≈ 0.0

# or minimal when only one probability is nonzero and equal to 1.0
@test information(Shannon(), Probabilities([1.0, 0.0, 0.0, 0.0])) ≈ 0.0
