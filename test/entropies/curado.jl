N = 10
bs = [0.5, 1.0, 2.0, 5.0, 10.0]

# Analytical tests from Curado (2004).
# -----------------------------------

# Curado entropy is maximized for uniform distribution.
max_hs = [entropy_maximum(Curado(b = b), N) for b in bs]
pu = Probabilities(repeat([1/N], N))
hs = [entropy(Curado(b = b), pu) for b in bs]
@test all([h ≈ maxh for (h, maxh) in zip(hs, max_hs)])

# Curado entropy is minimized and equal 0 for one-element distributions
p1 = Probabilities([1.0])
hs_p1 = [entropy(Curado(b = b), p1) for b in bs]
@test all([h ≈ 0.0 for h in hs_p1])
