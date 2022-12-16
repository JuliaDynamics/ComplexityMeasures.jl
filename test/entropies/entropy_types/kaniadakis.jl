N = 10
κs = [0.5, 1.0, 2.0, 5.0, 10.0]

probs = Probabilities([1, 3, 6,2, 9])

@test entropy(Kaniadakis(), probs) isa Real

@test entropy(Kaniadakis(), probs) >= 0

@test entropy(Kaniadakis(κ = 0, base = 2), probs) == entropy(Shannon(base = 2), probs)

# Kaniadakis does not state for which distribution for which Kaniadakis entropy is maximised