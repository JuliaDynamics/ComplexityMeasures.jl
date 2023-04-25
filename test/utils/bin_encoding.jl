using ComplexityMeasures, Test
using Random: MersenneTwister
@test TransferOperator(RectangularBinning(3)) isa TransferOperator


seeds = [1234, 57772, 90897, 2158081, 888]
# Note that if `false` is used for `precise` the tests will fail.
# But that's okay, since we do not do guarantees for that case.
binnings = [
    RectangularBinning(3, true),
    RectangularBinning(3, false),
    RectangularBinning(0.2, true),
    RectangularBinning(0.2, false),
    RectangularBinning([2, 3], true),
    RectangularBinning([2, 3], false),
    RectangularBinning([0.2, 0.3], true),
    FixedRectangularBinning(range(0, 1; length = 5), 2, true),
    FixedRectangularBinning(range(0, 1; length = 5), 2, false),
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

@testset "maxi in range corrected (###)" begin
    X = SVector{2, Float64}.(vec(collect(Iterators.product(0:0.05:0.99, 0:0.05:0.99))))
    X = StateSpaceSet(X)
    # From FractalDimensions.jl
    function _data_boxing(X, encoding)
        # Output is a dictionary mapping cartesian indices to vector of data point indices
        # in said cartesian index bin
        boxes_to_contents = Dict{NTuple{dimension(X), Int}, Vector{Int}}()
        for (j, x) in enumerate(X)
            i = encode(encoding, x) # linear index of box in histogram
            if i == -1
                error("$(j)-th point was encoded as -1. Point = $(x)")
            end
            ci = Tuple(encoding.ci[i]) # cartesian index of box in histogram
            if !haskey(boxes_to_contents, ci)
                boxes_to_contents[ci] = Int[]
            end
            push!(boxes_to_contents[ci], j)
        end
        return boxes_to_contents
    end

    @testset "0.1 rad" begin
        encoding = RectangularBinEncoding(RectangularBinning(0.1, true), X)
        btc = _data_boxing(X, encoding)
        @test all(isequal(4), length.(values(btc)))
    end
    @testset "0.05 rad" begin
        encoding = RectangularBinEncoding(RectangularBinning(0.05, true), X)
        btc = _data_boxing(X, encoding)
        @test all(isequal(1), length.(values(btc)))
    end
end