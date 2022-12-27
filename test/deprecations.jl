using Entropies, Test

x = randn(1000)

@test permentropy(x) == entropy_permutation(x; base=MathConstants.e)
@test genentropy(x, 0.1) == entropy(Shannon(MathConstants.e), ValueHistogram(0.1, x), x)
@test probabilities(x, 0.1) == probabilities(ValueHistogram(0.1, x), x)

x = Dataset(rand(100, 3))
@test genentropy(x, 4) == entropy(Shannon(MathConstants.e), ValueHistogram(4, x), x)