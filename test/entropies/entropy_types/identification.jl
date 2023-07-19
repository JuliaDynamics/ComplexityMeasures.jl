
# Analytical tests
# -----------------------------------
# Minimal and equal to zero when probability distribution has only one element...
@test entropy(Identification(), Probabilities([1.0])) ≈ 0.0

# or minimal when only one probability is nonzero and equal to 1.0
@test entropy(Identification(), Probabilities([1.0, 0.0, 0.0, 0.0])) ≈ 0.0

# Maximal for a uniform distribution
x = [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4]
@test entropy_normalized(Identification(), CountOccurrences(), x) ≈ 1.0

# Submaximal for a non-uniform distribution
y = [0.1, 0.2, 0.2, 0.25, 0.2, 0.351, 0.312, 0.3, 0.3, 0.4, 0.4]
@test entropy_normalized(Identification(), CountOccurrences(), y) < 1.0
