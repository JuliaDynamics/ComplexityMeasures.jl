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
        ]
        @testset "$(typeof(os[i]).name.name)" for i in eachindex(os)
            cts, Ω = counts_and_outcomes(os[i], x)
            @test length(cts) == length(Ω)

            cts = counts(os[i], x)
            @test cts isa Vector{Int}
        end
    end

    @testset "ND estimators" begin
        x = rand(30, 30)
        os = [
            SpatialDispersion([0 1; 1 0], x),
            SpatialSymbolicPermutation([0 1; 1 0], x),
        ]
        @testset "$(typeof(os[i]).name.name)" for i in eachindex(os)
            cts, Ω = counts_and_outcomes(os[i], x)
            @test length(cts) == length(Ω)

            cts = counts(os[i], x)
            @test cts isa Vector{Int}
        end
    end
end

@testset "Non-counting-based outcome spaces" begin
    x = rand(rng, 100)
    os = [
        WaveletOverlap(),
        TransferOperator(RectangularBinning(3)),
        PowerSpectrum(),
        SymbolicAmplitudeAwarePermutation(),
        SymbolicWeightedPermutation(),
        NaiveKernel(0.1),
    ]
    @testset "$(typeof(os[i]).name.name)" for i in eachindex(os)
        @test_throws ArgumentError counts_and_outcomes(os[i], x)
    end
end
