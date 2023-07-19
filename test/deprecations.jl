using ComplexityMeasures, Test

x = randn(1000)

@test permentropy(x) == entropy_permutation(x; base=MathConstants.e)
@test genentropy(x, 0.1) == information(Shannon(MathConstants.e), ValueHistogram(0.1), x)
@test probabilities(x, 0.1) == probabilities(ValueHistogram(0.1), x)

x = StateSpaceSet(rand(100, 3))
@test genentropy(x, 4) == information(Shannon(MathConstants.e), ValueHistogram(4), x)

@test entropy(Shannon(MathConstants.e), ValueHistogram(4), x) ==
    information(Shannon(MathConstants.e), ValueHistogram(4), x)

@test entropy_maximum(Shannon(MathConstants.e), ValueHistogram(4), x) ==
    information_maximum(Shannon(MathConstants.e), ValueHistogram(4), x)

@test entropy_normalized(Shannon(MathConstants.e), ValueHistogram(4), x) ==
    information_normalized(Shannon(MathConstants.e), ValueHistogram(4), x)
