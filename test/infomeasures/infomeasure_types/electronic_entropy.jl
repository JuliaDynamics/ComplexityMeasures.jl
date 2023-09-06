using Test, ComplexityMeasures

h = Shannon(; base = 2); j = ShannonExtropy(; base = 3)
@test_throws ArgumentError ElectronicEntropy(; h, j)
# Analytical tests
# -----------------------------------
# Minimal and equal to zero when probability distribution has only one element...
@test information(ElectronicEntropy(), Probabilities([1.0])) ≈ 0.0

# or minimal when only one probability is nonzero and equal to 1.0
@test information(ElectronicEntropy(), Probabilities([1.0, 0.0, 0.0, 0.0])) ≈ 0.0

# Minimized for one-element distributions, where there is total order.
x = [0.1, 0.1, 0.1]
est = UniqueElements()
@test information(ElectronicEntropy(), est, x) == 0.0
@test information_normalized(ElectronicEntropy(), est, x) == 0.0

# Maximized for a uniform distribution
x = [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]
@test information_normalized(ElectronicEntropy(), est, x) ≈ 1.0

# Submaximal for an onuniform distribution
x = [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4]
@test information_normalized(ElectronicEntropy(), est, x) < 1.0

# Maximum is equal to the maxima of the entropy and extropy.
e = ElectronicEntropy()
@test information_maximum(e, 5) == information_maximum(e.h, 5) + information_maximum(e.j, 5)
