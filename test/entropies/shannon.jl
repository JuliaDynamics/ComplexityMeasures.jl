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
