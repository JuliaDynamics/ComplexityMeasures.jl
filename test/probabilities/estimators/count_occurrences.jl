# Analytic tests: Booleans have two possible states
# counting enough of them should give each with equal probability
using Entropies
using Random, Test
rng = Random.MersenneTwister(1234)
x = [rand(rng, Bool) for _ in 1:10000]

probs1 = probabilities(x)
probs2 = probabilities(CountOccurrences(x), x)
probs3, outs = probabilities_and_outcomes(CountOccurrences(x), x)

for ps in (probs1, probs2, probs3)
    for p in ps; @test 0.49 < p < 0.51; end
end

@test sort(outs) == [false, true]
@test sort(outcome_space(CountOccurrences(x))) == [false, true]
@test total_outcomes(CountOccurrences(x)) == 2

# Same for 2D sets (outcomes not tested here)
y = [rand(rng, Bool) for _ in 1:10000]
D = Dataset(x, y)

probs1 = probabilities(D)
probs2 = probabilities(CountOccurrences(D), D)
for ps in (probs1, probs2)
    for p in ps; @test 0.24 < p < 0.26; end
end

# Renyi of coin toss is 1 bit, and for two coin tosses is two bits
# Result doesn't depend on `q` due to uniformity of the PDF.
for q in (0.5, 1.0, 2.0)
    h = entropy(Renyi(q), CountOccurrences(x), x)
    @test 0.99 < h < 1.01
    h = entropy(Renyi(q), CountOccurrences(D), D)
    @test 1.99 < h < 2.01
end
