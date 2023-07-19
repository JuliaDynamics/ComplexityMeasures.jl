N = 10
κs = [0.5, 1.0, 2.0, 5.0, 10.0]

probs = Probabilities([1, 3, 6,2, 9])
@test information(Kaniadakis(), probs) isa Real
@test information(Kaniadakis(), probs) >= 0
@test information(Kaniadakis(κ = 0, base = 2), probs) == information(Shannon(base = 2), probs)

p = Probabilities([0, 1])
@test information(Kaniadakis(), p) == 0

# Kaniadakis does not state for which distribution for which Kaniadakis entropy is maximised
