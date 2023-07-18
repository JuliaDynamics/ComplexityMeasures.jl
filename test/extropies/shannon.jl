

# Minimized for one-element distributions, where there is total order.
x = [0.1, 0.1, 0.1]
@test extropy(ShannonExtropy(), CountOccurrences(), x) == 0.0
@test extropy_normalized(ShannonExtropy(), CountOccurrences(), x) == 0.0

# Normalized Shannon extropy should be maximized (i.e. be equal to 1) for a
# uniform distribution.
x = [0.2, 0.2, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6]
js = extropy_normalized(ShannonExtropy(), CountOccurrences(), x)
@test js ≈ 1

# It should be less than 1 for non-uniform distributions
x = [0.2, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6]
js = extropy_normalized(ShannonExtropy(), CountOccurrences(), x)
@test js < 1

# Correctness of maximum
x = [0.2, 0.2, 0.4, 0.4, 0.5, 0.5]
js = extropy(ShannonExtropy(base = 2), CountOccurrences(), x)
@test js ≈ (3 - 1)*log(2, (3 / (3 - 1)))
