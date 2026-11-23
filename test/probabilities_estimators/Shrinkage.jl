using ComplexityMeasures
using Random
using Test
rng = MersenneTwister(1234)
@testset "Shrinkage: counting-based outcome space" begin
    @testset "1D estimators" begin
        x = rand(rng, 1:10.0, 100)

        os = [
            UniqueElements(),
            OrdinalPatterns(m = 3),
            Dispersion(),
            CosineSimilarityBinning(),
            ValueBinning(RectangularBinning(3)),
        ]
        @testset "$(nameof(typeof(os[i])))" for i in eachindex(os)
            est = Shrinkage()
            outcomemodel = os[i]

            ps, Ωobs = probabilities_and_outcomes(est, outcomemodel, x)
            @test ps isa Probabilities
            @test probabilities(est, outcomemodel, x) == ps
            @test outcomes(est, outcomemodel, x) == Ωobs

            ps, Ωall = allprobabilities_and_outcomes(est, outcomemodel, x)
            @test ps isa Probabilities
            @test sort(Ωall) == outcome_space(est, outcomemodel, x)
            @test allprobabilities(est, outcomemodel, x) isa Probabilities
            @test sort(outcome_space(est, outcomemodel, x)) == sort(Ωall)
        end

        # Spatial estimators (all of them are currently counting-based)
        # -------------------------------------------------------------
        x = rand(50, 50)
        os = [
            SpatialDispersion([0 1; 1 0], x, c = 2),
            SpatialOrdinalPatterns([0 1; 1 0], x),
        ]
        @testset "$(nameof(typeof(os[i])))" for i in eachindex(os)
            est = Shrinkage()
            outcomemodel = os[i]
            ps, Ωobs = probabilities_and_outcomes(est, outcomemodel, x)
            @test ps isa Probabilities
            @test outcomes(est, outcomemodel, x) == Ωobs
            @test probabilities(est, outcomemodel, x) == ps

            ps, Ωall = allprobabilities_and_outcomes(est, outcomemodel, x)
            @test ps isa Probabilities
            @test sort(Ωall) == sort(outcome_space(est, outcomemodel, x))
            @test allprobabilities(est, outcomemodel, x) == ps
        end
    end
end

# We don't need to test for non-counting based outcome spaces, because the
# Shrinkage constructor prevents us from combining such outcome spaces with
# the estimator.

@testset "Shrinkage: optimal shrinkage intensity" begin
    # Counts 6, 3, 1 (n = 10), shrunk towards the uniform target t = 1/3. Optimal
    # intensity, eq. 5 in Hausser & Strimmer (2009):
    #     λ = (1 - Σ θ̂ₖ²) / ((n - 1) Σ (tₖ - θ̂ₖ)²)
    x = vcat(fill(1, 6), fill(2, 3), fill(3, 1))
    θ̂ = [0.6, 0.3, 0.1]
    t = 1 / 3
    λ = (1 - sum(θ̂ .^ 2)) / ((10 - 1) * sum((t .- θ̂) .^ 2))
    ps = probabilities(Shrinkage(), UniqueElements(), x)
    @test collect(ps) ≈ λ .* t .+ (1 - λ) .* θ̂

    # A single observed outcome equals the target, so nothing is shrunk.
    @test collect(probabilities(Shrinkage(), UniqueElements(), fill(1, 10))) == [1.0]
end
