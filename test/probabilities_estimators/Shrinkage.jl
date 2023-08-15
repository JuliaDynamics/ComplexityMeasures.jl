using Random
using Test
rng = MersenneTwister(1234)
@testset "Counting-based outcome space" begin
    @testset "1D estimators" begin
        x = rand(rng, 1:10., 100)

        os = [
            CountOccurrences(),
            SymbolicPermutation(m = 3),
            Dispersion(),
            ValueHistogram(RectangularBinning(3)),
            NaiveKernel(0.1),
        ]
        @testset "$(typeof(os[i]).name.name)" for i in eachindex(os)
            est = Shrinkage(os[i])
            ps, Ωobs = probabilities_and_outcomes(est, x)
            @test ps isa Probabilities

            ps, Ωall = allprobabilities_and_outcomes(est, x)
            @test ps isa Probabilities
            @test sort(Ωall) == outcome_space(est, x)
        end
    end

    @testset "ND estimators" begin
        x = rand(50, 50)
        os = [
            SpatialDispersion([0 1; 1 0], x, c = 2),
            SpatialSymbolicPermutation([0 1; 1 0], x),
        ]
        @testset "$(typeof(os[i]).name.name)" for i in eachindex(os)
            est = Shrinkage(os[i])
            ps, Ωobs = probabilities_and_outcomes(est, x)
            @test ps isa Probabilities

            ps, Ωall = allprobabilities_and_outcomes(est, x)
            @test ps isa Probabilities
            @test sort(Ωall) == sort(outcome_space(est, x))
        end
    end
end

# If counting isn't well-defined for the outcome space, then all relevant functions
# just return an argument error.
@testset "Non-counting-based outcome spaces" begin
    x = rand(rng, 100)
    os = [
        WaveletOverlap(),
        TransferOperator(RectangularBinning(3)),
        PowerSpectrum(),
        SymbolicAmplitudeAwarePermutation(),
        SymbolicWeightedPermutation(),
    ]
    @testset "$(typeof(os[i]).name.name)" for i in eachindex(os)
        est = Shrinkage(os[i])
        ps, Ωobs = probabilities_and_outcomes(est, x)
        @test ps isa Probabilities

        ps, Ωall = allprobabilities_and_outcomes(est, x)
        @test ps isa Probabilities
        @test sort(Ωall) == sort(outcome_space(est, x))
    end
end
