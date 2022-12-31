using ComplexityMeasures, Test

# Interface tests
x = rand(1000)
xp = Probabilities(x)
@test_throws MethodError entropy(Renyi(q = 2), x)
@test entropy(Renyi(q = 2), xp) isa Real
@test entropy(Renyi(q = 1), xp) isa Real

@test entropy_maximum(Renyi(), 2) == 1

# Analytical tests
for q in (0, 0.5, 1, 2.0) # independent of q
    # Minimal and equal to zero when probability distribution has only one element...
    @test entropy(Renyi(q), Probabilities([1.0])) ≈ 0.0
    # or minimal when only one probability is nonzero and equal to 1.0
    @test entropy(Renyi(q), Probabilities([1.0, 0.0, 0.0, 0.0])) ≈ 0.0
end