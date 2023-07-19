p = Probabilities(repeat([1/5], 5))
@test round(information(Tsallis(q = -1/2, k = 1), p), digits = 2) ≈ 6.79

# Analytical tests from Tsallis (1998)
# -----------------------------------
# Probability distribution has only  one element
@test information(Tsallis(), Probabilities([1.0])) ≈ 0.0

# One and only one probability of 1.0 => entropy → 0.
@test information(Tsallis(), Probabilities([1.0, 0.0, 0.0, 0.0])) ≈ 0.0

# Uniform distribution maximizes Tsallis entropy, which then equals log(N),
# where N is the number of states. Then the entropy attains its information_maximum value
# (N^(1 - q) - 1) / (1 - q)
N = 4
ps = Probabilities(repeat([1/N], N))

q_cases = [-2.0, -0.5, 0.5, 2.0]
t_entropies = [information(Tsallis(q = q, k = 1), ps) for q in q_cases]
maxvals = [information_maximum(Tsallis(q = q, k = 1), N) for q in q_cases]
@test all(t_entropies .≈ maxvals)

q_cases = [-2.0, -0.5, 0.5, 2.0]
k = 2
t_entropies = [information(Tsallis(q = q, k = k), ps) for q in q_cases]
maxvals = [information_maximum(Tsallis(q = q, k = k), N) for q in q_cases]
@test all(t_entropies .≈ maxvals)

# Reduces to Shannon entropy for q → 1.0
@test information(Tsallis(base = 2, q = 1.0), ps) ≈ log(2, N)
