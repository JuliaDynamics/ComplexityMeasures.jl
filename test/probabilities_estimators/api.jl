# Counting-basedoutcome spaces work with any estimator
os = [
    CountOccurrences(),
    SymbolicPermutation(m = 3),
    Dispersion(),
    Diversity(),
    ValueHistogram(RectangularBinning(3)),
]
@testset "`ProbabilitiesEstimator` constructors: $(typeof(os[i]).name.name)" for i in eachindex(os)
    @test RelativeAmount(os[i]) isa RelativeAmount
    @test Bayes(os[i]) isa Bayes
    @test Shrinkage(os[i]) isa Shrinkage
    @test AddConstant(os[i]) isa AddConstant
end

# Non-counting based outcome spaces don't work with counting-based estimators
os = [
    WaveletOverlap(),
    TransferOperator(RectangularBinning(3)),
    PowerSpectrum(),
    SymbolicAmplitudeAwarePermutation(),
    SymbolicWeightedPermutation(),
    NaiveKernel(0.1),
]
@testset "`ProbabilitiesEstimator` constructors: $(typeof(os[i]).name.name)" for i in eachindex(os)
    @test RelativeAmount(os[i]) isa RelativeAmount
    @test_throws ArgumentError Bayes(WaveletOverlap())
    @test_throws ArgumentError Shrinkage(WaveletOverlap())
    @test_throws ArgumentError AddConstant(WaveletOverlap())
end