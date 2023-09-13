N = 10
bs1 = [0.5, 1.0, 2.0, 5.0, 10.0]

# Analytical tests from Curado (2004).
# -----------------------------------

# Curado entropy is maximized for uniform distribution.
max_hs = [information_maximum(Curado(b = b), N) for b in bs1]
pu = Probabilities(repeat([1/N], N))
hs = [information(Curado(b = b), pu) for b in bs1]
@test all([h ≈ maxh for (h, maxh) in zip(hs, max_hs)])

# Curado entropy is minimized and equal 0 for one-element distributions
p1 = Probabilities([1.0])
hs_p1 = [information(Curado(b = b), p1) for b in bs1]
@test all([h ≈ 0.0 for h in hs_p1])


N1 = 20
bs2 = [0.2, 0.4, 0.6, 0.8, 1.0]

max_hs_modified = [information_maximum(Curado(b = b), N1) for b in bs2]
pu_modified = Probabilities(repeat([1/N1], N1))
hs_modified = [information(Curado(b = b), pu_modified) for b in bs2]
@test all([h ≈ maxh for (h, maxh) in zip(hs_modified, max_hs_modified)])

p1_modified = Probabilities([0.85])
hs_p1_modified = [information(Curado(b = b), p1_modified) for b in bs2]
@test all([h ≈ 0.0 for h in hs_p1_modified])