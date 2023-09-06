using Test
using Random
rng = Xoshiro(1234)

@testset "`ProbabilitiesEstimator` constructors" begin
    @test RelativeAmount() isa RelativeAmount
    @test BayesianRegularization() isa BayesianRegularization
    @test Shrinkage() isa Shrinkage
    @test AddConstant() isa AddConstant
end

# Counting-basedoutcome spaces work with any estimator
os_count = [
    UniqueElements(),
    OrdinalPatterns(m = 3),
    Dispersion(),
    Diversity(),
    ValueBinning(RectangularBinning(3)),
]

x = rand(rng, 100)
@testset "`ProbabilitiesEstimator` with non-count-based : $(typeof(os_count[i]).name.name)" for i in eachindex(os_count)
    o = os_count[i]
    @test probabilities(RelativeAmount(), o, x) isa Probabilities
    @test probabilities(BayesianRegularization(), o, x) isa Probabilities
    @test probabilities(Shrinkage(), o, x) isa Probabilities
    @test probabilities(AddConstant(), o, x) isa Probabilities
end


# Non-counting based outcome spaces don't work with counting-based estimators
os_noncount = [
    WaveletOverlap(),
    TransferOperator(RectangularBinning(3)),
    PowerSpectrum(),
    AmplitudeAwareOrdinalPatterns(),
    WeightedOrdinalPatterns(),
    NaiveKernel(0.1)
]

@testset "`ProbabilitiesEstimator` constructors: $(typeof(os_noncount[i]).name.name)" for i in eachindex(os_noncount)
    o = os_noncount[i]
    # RelativeAmount is the catch-all estimator that also works for pseudo counts
    @test probabilities(RelativeAmount(), o, x) isa Probabilities

    # Count-based estimators should error.
    @test_throws ArgumentError probabilities(BayesianRegularization(), o, x)
    @test_throws ArgumentError probabilities(Shrinkage(), o, x)
    @test_throws ArgumentError probabilities(AddConstant(), o, x)
end
