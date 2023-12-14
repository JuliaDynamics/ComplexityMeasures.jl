using ComplexityMeasures
using Random
using Test
rng = MersenneTwister(1234)

@testset "RelativeAmount" begin
    @testset "1D estimators" begin
        x = rand(rng, 1:10., 100)

        os = [
            UniqueElements(),
            OrdinalPatterns(m = 3),
            Dispersion(),
            CosineSimilarityBinning(),
            ValueBinning(RectangularBinning(3)),
        ]

        @testset "$(nameof(typeof(os[i])))" for i in eachindex(os)
            est = RelativeAmount()
            o = os[i]

            cts, Ωobs = allcounts_and_outcomes(o, x)
            @test sort(Ωobs) == sort(outcome_space(o, x))
            @test length(cts) == total_outcomes(o, x)
            @test allcounts(o, x) == cts

            cts, Ωobs = counts_and_outcomes(o, x)
            @test cts isa Counts{<:Integer, 1}
            @test length(cts) <= total_outcomes(o, x)
            @test length(Ωobs) <= total_outcomes(o, x)
            @test counts(o, x) == cts

            ps, Ωobs = probabilities_and_outcomes(est, o, x)
            @test ps isa Probabilities
            @test probabilities(o, x) == ps
            @test outcomes(o, x) == Ωobs

            ps, Ωall = allprobabilities_and_outcomes(est, o, x)
            @test ps isa Probabilities
            @test sort(Ωall) == outcome_space(o, x)
            @test allprobabilities(o, x) isa Probabilities
            @test sort(outcome_space(o, x)) == sort(Ωall)
        end

        # Spatial estimators (all of them are currently counting-based)
        # -------------------------------------------------------------
        x = rand(50, 50)
        os = [
            SpatialDispersion([0 1; 1 0], x, c = 2),
            SpatialOrdinalPatterns([0 1; 1 0], x),
        ]
        @testset "$(nameof(typeof(os[i])))" for i in eachindex(os)
            est = RelativeAmount()
            o = os[i]
            ps, Ωobs = probabilities_and_outcomes(est, o, x)
            @test ps isa Probabilities
            @test outcomes(o, x) == Ωobs
            @test probabilities(o, x) == ps

            ps, Ωall = allprobabilities_and_outcomes(est, o, x)
            @test ps isa Probabilities
            @test sort(Ωall) == sort(outcome_space(o, x))
            @test allprobabilities(o, x) == ps
        end
    end
end

# If counting isn't well-defined for the outcome space, then all relevant functions
# just return an argument error.
@testset "RelativeAmount: Non-counting-based outcome spaces" begin
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
        est = RelativeAmount()
        o = os[i]
        ps, Ωobs = probabilities_and_outcomes(o, x)
        @test ps isa Probabilities
        @test outcomes(o, x) == Ωobs
        @test probabilities(o, x) ≈ ps

        ps, Ωall = allprobabilities_and_outcomes(o, x)
        @test ps isa Probabilities
        @test sort(Ωall) == sort(outcome_space(o, x))

        # `TransferOperator` uses randomization, so exact comparison sometimes fails
        @test allprobabilities(o, x) ≈ ps
    end
end
