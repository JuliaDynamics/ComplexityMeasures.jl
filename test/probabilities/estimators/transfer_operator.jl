using StateSpaceSets: Dataset

@test TransferOperator(RectangularBinning(3)) isa TransferOperator

D = Dataset(rand(100, 2))

binnings = [
    RectangularBinning(3),
    RectangularBinning(0.2),
    RectangularBinning([2, 3]),
    RectangularBinning([0.2, 0.3]),
]

# There's not easy way of constructing an analytical example for the resulting
# probabilities, because they are computed according to the approximation to the
# transfer operator, not as naive visitation freqiencies to the bins. We just test
# generically here.
# TODO: make a stupidly simple example where we can actually compute the measure of
# each bin exactly.
fb = FixedRectangularBinning(0, 1, 2)
@test total_outcomes(D, fb) == 2^2
@test outcome_space(D, fb) ≈ SVector.([(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)])

# For non-fixed binnings, an error should be thrown for both `total_outcomes` and
# `outcome_space`.
@testset "Binning test $i" for i in eachindex(binnings)
    @test_throws ErrorException total_outcomes(D, binnings[i]) == 2^2
    @test_throws ErrorException outcome_space(D, binnings[i]) == 2^2

end

@testset "Binning test $i" for i in eachindex(binnings)
    to = Entropies.transferoperator(D, binnings[i])
    @test to isa Entropies.TransferOperatorApproximationRectangular

    iv = invariantmeasure(to)
    @test iv isa InvariantMeasure

    p, bins = invariantmeasure(iv)
    @test p isa Probabilities
    @test bins isa Vector{<:SVector}

    est = TransferOperator(binnings[i])
    @test probabilities(est, D) isa Probabilities
    @test probabilities_and_outcomes(est, D) isa Tuple{Probabilities, Vector{SVector{2, Float64}}}
end
