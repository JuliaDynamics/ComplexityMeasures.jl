using ComplexityMeasures, Test
using Random: MersenneTwister
@test TransferOperator(RectangularBinning(3)) isa TransferOperator


seeds = [1234, 57772, 90897, 2158081, 888]
# Note that if `false` is used for `precise` the tests will fail.
# But that's okay, since we do not do guarantees for that case.
binnings = [
    RectangularBinning(3, true),
    RectangularBinning(0.2, true),
    RectangularBinning([2, 3], true),
    RectangularBinning([0.2, 0.3], true),
    FixedRectangularBinning(range(0, 1; length = 5), 2, true)
]

@testset "seed $(seed)" for seed in seeds
    D = StateSpaceSet(rand(MersenneTwister(seed), 100, 2))

    @testset "Binning $i" for i in eachindex(binnings)
        encoding = RectangularBinEncoding(binnings[i], D)
        outs = [encode(encoding, χ) for χ in D]
        unique!(outs)
        @test -1 ∉ unique!(outs)
        decs = [decode(encoding, i) for i in outs]

    end
end
