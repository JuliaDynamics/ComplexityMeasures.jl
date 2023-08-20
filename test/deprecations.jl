using ComplexityMeasures, Test

x = randn(1000)

# Convenience
@test permentropy(x) == entropy_permutation(x; base=MathConstants.e)
msg ="`permentropy(x; τ, m, base)` is deprecated.\nUse instead: `entropy_permutation(x; τ, m, base)`, or even better, use the\ndirect syntax discussed in the docstring of `entropy_permutation`.\n"
@test_logs (:warn, msg) permentropy(x)

# Generalized entropies
@test genentropy(x, 0.1) == information(Shannon(MathConstants.e), ValueHistogram(0.1), x)
msg = "`genentropy(probs::Probabilities; q, base)` deprecated.\nUse instead: `information(Renyi(q, base), probs)`.\n"
@test_logs (:warn, msg) genentropy(Probabilities(rand(3)))

msg = "`genentropy(x::Array_or_SSSet, est::ProbabilitiesEstimator; q, base)` is deprecated.\nUse instead: `information(Renyi(q, base), est, x)`.\n"
@test_logs (:warn, msg) genentropy(x, ValueHistogram(0.1))

@test probabilities(x, 0.1) == probabilities(ValueHistogram(0.1), x)
msg = "`probabilities(x, est::OutcomeSpace)`\nis deprecated, use `probabilities(est::OutcomeSpace, x) instead`.\n"
@test_logs (:warn, msg) probabilities(x, ValueHistogram(0.1))

x = StateSpaceSet(rand(100, 3))
@test genentropy(x, 4) == information(Shannon(MathConstants.e), ValueHistogram(4), x)


@testset "deprecations: binning" begin
    @test FixedRectangularBinning((0, 1), (0, 1), 2) isa FixedRectangularBinning
    @test FixedRectangularBinning(0.1, 0.9, 2) isa FixedRectangularBinning
    @test FixedRectangularBinning(0.1, 0.9, 0.2) isa FixedRectangularBinning
end


@testset "2.9 deprecations" begin
    # For

    @test entropy_maximum(Shannon(MathConstants.e), ValueHistogram(4), x) ==
        information_maximum(Shannon(MathConstants.e), ValueHistogram(4), x)

    @test entropy_normalized(Shannon(MathConstants.e), ValueHistogram(4), x) ==
        information_normalized(Shannon(MathConstants.e), ValueHistogram(4), x)

    # Providing information measure as first argument shouldn't work, but does so only for
    # Shannon, for backwards compatibility.
    @test entropy(Shannon(), Kraskov(), x) isa Real
    @test_throws ErrorException entropy(Tsallis(), Kraskov(), x)

    msg = "`entropy(est::DifferentialEntropyEstimator, x)` is deprecated.\nUse `information(est, x)` instead.\n"
    @test_logs (:warn, msg) entropy(Kraskov(), x)

    @test SymbolicPermutation() isa OrdinalPatterns
    @test SymbolicWeightedPermutation() isa WeightedOrdinalPatterns
    @test SymbolicAmplitudeAwarePermutation() isa AmplitudeAwareOrdinalPatterns
end
