@test TransferOperator(RectangularBinning(3)) isa TransferOperator

D = Dataset(rand(1000, 3))

binnings = [
    RectangularBinning(3),
    RectangularBinning(0.2),
    RectangularBinning([2, 2, 3]),
    RectangularBinning([0.2, 0.3, 0.3])
]

@testset "Binning test $i" for i in eachindex(binnings)
    to = Entropies.transferoperator(D, binnings[i])
    @test to isa Entropies.TransferOperatorApproximationRectangular

    iv = invariantmeasure(to)
    @test iv isa InvariantMeasure

    p, bins = invariantmeasure(iv)
    @test p isa Probabilities
    @test bins isa Vector{<:SVector}

    @test probabilities(D, TransferOperator(binnings[i])) isa Probabilities
end
