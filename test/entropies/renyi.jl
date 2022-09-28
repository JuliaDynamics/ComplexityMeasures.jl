x = rand(1000)
xp = Probabilities(x)
@test_throws MethodError entropy(Renyi(q = 2), x) isa Real
@test entropy(Renyi(q = 2), xp) isa Real
@test entropy(Renyi(q = 1), xp) isa Real

# Analytical tests
# -----------------------------------
# Minimal and equal to zero when probability distribution has only one element...
@test entropy(Renyi(q = 1), Probabilities([1.0])) ≈ 0.0
@test entropy(Renyi(q = 0.5), Probabilities([1.0])) ≈ 0.0
@test entropy(Renyi(q = 2.0), Probabilities([1.0])) ≈ 0.0

# or minimal when only one probability is nonzero and equal to 1.0
@test entropy(Renyi(q = 0.5), Probabilities([1.0, 0.0, 0.0, 0.0])) ≈ 0.0
@test entropy(Renyi(q = 1.0), Probabilities([1.0, 0.0, 0.0, 0.0])) ≈ 0.0
@test entropy(Renyi(q = 2.0), Probabilities([1.0, 0.0, 0.0, 0.0])) ≈ 0.0
