using Random
using Test
rng = MersenneTwister(1234)

@testset "interface" begin
    x = ones(3)
    p = probabilities(x)
    @test p isa Probabilities
    @test p == [1]
end

@testset "Histogram" begin
    # Okay, with these inputs we know exactly what the final scenario
    # should be. Bins 1 2 and 3 have one entry, bin 4 has zero, bin 5
    # has two entries, bins 6 7 8 have zero, bin 9 has 1 and bin 10 zero
    x = [0.05, 0.15, 0.25, 0.45, 0.46, 0.85]
    o = ValueBinning(FixedRectangularBinning((0:0.1:1,)))
    correct = [1, 1, 1, 0, 2, 0, 0, 0, 1, 0]
    correct = correct ./ sum(correct)

    miss = missing_outcomes(o, x)
    @test miss == 5

    p = allprobabilities(o, x)

    @test p ≈ correct
end

@testset "Ordinal" begin
    # here it is guaranteed that all outcomes are the 123 permutation
    # except the final one that is the 312. That permutation is
    # fourth in the order of the ordinal patterns for m=3
    x = [1, 2, 3, 4, 5, 6, 1]
    o = OrdinalPatterns(m = 3)
    probs, outs = probabilities_and_outcomes(o, x)
    correct = [4, 0, 0, 0, 1, 0]
    correct = correct ./ sum(correct)
    p = allprobabilities(o, x)
    @test p == correct

    miss = missing_outcomes(o, x)
    @test miss == 4

    # Make sure that different outcome would get different allprobs
    x = [1, 2, 3, 4, 5, 6, 5.5]
    probs, outs = probabilities_and_outcomes(o, x)
    correct = [4, 1, 0, 0, 0, 0]
    correct = correct ./ sum(correct)
    p = allprobabilities(o, x)
    @test p == correct
end

@testset "Counting-based outcome space" begin

    @testset "1D estimators" begin
        x = rand(rng, 1:10., 100)

        os = [
            UniqueElements(),
            OrdinalPatterns(m = 3),
            Dispersion(),
            ValueBinning(RectangularBinning(3)),
        ]
        @testset "$(nameof(typeof(os[i])))" for i in eachindex(os)
            @test typeof(os[i]) <: ComplexityMeasures.CountBasedOutcomeSpace
            cts, Ω = counts_and_outcomes(os[i], x)
            @test length(cts) == length(Ω)

            cts = counts(os[i], x)
            @test cts isa Counts
        end
    end

    @testset "ND estimators" begin
        x = rand(30, 30)
        os = [
            SpatialDispersion([0 1; 1 0], x),
            SpatialOrdinalPatterns([0 1; 1 0], x),
        ]
        @testset "$(nameof(typeof(os[i])))" for i in eachindex(os)
            @test typeof(os[i]) <: ComplexityMeasures.CountBasedOutcomeSpace
            cts, Ω = counts_and_outcomes(os[i], x)
            @test cts isa Counts
            @test length(cts) == length(Ω)
        end
    end
end

@testset "Non-counting-based outcome spaces" begin
    x = rand(rng, 100)
    os = [
        WaveletOverlap(),
        TransferOperator(RectangularBinning(3)),
        PowerSpectrum(),
        AmplitudeAwareOrdinalPatterns(),
        WeightedOrdinalPatterns(),
        NaiveKernel(0.1),
    ]
    @testset "$(nameof(typeof(os[i])))" for i in eachindex(os)
        @test !ComplexityMeasures.is_counting_based(os[i])
        @test_throws "`counts`" counts_and_outcomes(os[i], x)
    end
end
