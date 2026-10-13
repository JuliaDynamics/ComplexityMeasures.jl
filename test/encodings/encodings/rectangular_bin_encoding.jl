using ComplexityMeasures, Test
using Random: MersenneTwister
using StateSpaceSets: StateSpaceSet
using StaticArrays: SVector

@testset "RectangularBinEncoding" begin
    x = rand(100)
    y = StateSpaceSet(rand(100, 2))
    b = FixedRectangularBinning(0:0.2:1.0)
    @test RectangularBinEncoding(b, x) isa RectangularBinEncoding
    @test_throws ArgumentError RectangularBinEncoding(b, y)
    @test_throws ArgumentError FixedRectangularBinning((5:-1:0,))
end

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


# TODO: the following tests were commented out at some point. Are they now redunant, given
# the tests above?

# @testset "Equal-width intervals (rectangular binning)" begin
#     N = 10 # each dimension is divides [min, max] into N chunks
#     @testset "User-defined grid" begin
#         # For datasets
#         # --------------------------------
#         D = StateSpaceSet([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0])
#         r = (1 - (-1)) / N

#         # (-1, 1) range for all dims.
#         binning_same = FixedRectangularBinning(-1.0, 1.0, N)

#         # (axismins[1], axismaxs[1]) gives the range for the 1st dim.
#         axismins, axismaxs = (-1.0, -1.0), (1.0, 1.0)
#         binning_diff = FixedRectangularBinning(axismins, axismaxs, N)

#         # Verify that grids are as expected.
#         symb_diff = RectangularBinEncoding(D, binning_diff)
#         symb_same = RectangularBinEncoding(D, binning_same)
#         @test all(@. symb_diff.edgelengths ≈ r)
#         @test all(@. symb_same.edgelengths ≈ r)
#         @test all(@. symb_diff.mini ≈ -1.0)
#         @test all(@. symb_same.mini ≈ -1.0)

#         # bins are indexed from 0, so should get the following symbols
#         expected = [SVector(0,0), SVector(4, 4), SVector(9, 9)]
#         @test outcomes(D, symb_diff) ==
#             outcomes(D, symb_same) ==
#             expected

#         @test total_outcomes(symb_diff) == N^2
#         @test total_outcomes(symb_same) == N^2
#         @test total_outcomes(D, symb_diff) == N^2
#         @test total_outcomes(D, symb_same) == N^2

#         # all possible outcomes
#         o = outcome_space(symb_diff)
#         @test length(o) == N^2
#         @test eltype(o) == SVector{2, Int}
#         for i in 1:N
#             for j in 1:N
#                 @test SVector(i, j) ∈ o
#             end
#         end

#         # For univariate timeseries
#         # --------------------------------
#         # bins are indexed from 0, so should get [0, 4, 9] with N = 10
#         x = [-1.0, 0.0, 1.0]
#         symb_1D = RectangularBinEncoding(x, binning_same)

#         @test total_outcomes(symb_1D) == N
#         @test total_outcomes(x, symb_1D) == N
#         @test symb_1D.mini ≈ -1.0
#         @test symb_1D.edgelengths ≈ (1 - (-1)) / N
#         @test outcomes(x, symb_1D) == [0, 4, 9]
#     end

#     @testset "Grid defined by data" begin
#         D = StateSpaceSet([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0])
#         r = nextfloat((1.0 - (-1.0)) / N)
#         binning_int = RectangularBinning(N)
#         binning_intvec = RectangularBinning([N, N])
#         binning_float = RectangularBinning(r)
#         binning_floatvec = RectangularBinning([r, r])

#         symb_int = RectangularBinEncoding(D, binning_int)
#         symb_float = RectangularBinEncoding(D, binning_float)
#         symb_intvec = RectangularBinEncoding(D, binning_intvec)
#         symb_floatvec = RectangularBinEncoding(D, binning_floatvec)

#         @test round.(symb_int.edgelengths, digits = 15) ==
#             round.(symb_float.edgelengths, digits = 15) ==
#             round.(symb_intvec.edgelengths, digits = 15) ==
#             round.(symb_floatvec.edgelengths, digits = 15)
#         @test all(@. symb_int.edgelengths ≈ r)
#         @test all(@. symb_float.edgelengths ≈ r)
#         @test all(@. symb_intvec.edgelengths ≈ r)
#         @test all(@. symb_floatvec.edgelengths ≈ r)
#         @test all(@. symb_int.mini ≈ -1.0)
#         @test all(@. symb_float.mini ≈ -1.0)
#         @test all(@. symb_intvec.mini ≈ -1.0)
#         @test all(@. symb_floatvec.mini ≈ -1.0)

#         @test outcomes(D, symb_int) ==
#             outcomes(D, symb_intvec) ==
#             outcomes(D, symb_float) ==
#             outcomes(D, symb_floatvec) ==
#             [SVector(0,0), SVector(4,4), SVector(9,9)]

#         x = [-1.0, 0.0, 1.0]
#         r = (1.0 - (-1.0)) / N
#         binning_int = RectangularBinning(N)
#         binning_intvec = RectangularBinning([N])
#         symb_int = RectangularBinEncoding(x, binning_int)
#         symb_float = RectangularBinEncoding(x, binning_float)

#         @test all(@. symb_int.edgelengths ≈ r)
#         @test all(@. symb_float.edgelengths ≈ r)
#         @test all(@. symb_int.mini ≈ -1.0)
#         @test all(@. symb_float.mini ≈ -1.0)

#         @test outcomes(x, symb_int) ==
#             outcomes(x, symb_float) ==
#             [0, 4, 9]
#     end

#     @testset "outcome length" begin
#         X = StateSpaceSet(rand(10, 3))
#         x = rand(10)
#         rbN = RectangularBinning(5)
#         rbNs = RectangularBinning([5, 3, 4])
#         rbF = RectangularBinning(0.5)
#         rbFs = RectangularBinning([0.5, 0.4, 0.3])

#         symbolization_xN = RectangularBinEncoding(x, rbN)
#         symbolization_xF = RectangularBinEncoding(x, rbF)
#         symbolization_XN = RectangularBinEncoding(X, rbN)
#         symbolization_XNs = RectangularBinEncoding(X, rbNs)
#         symbolization_XF = RectangularBinEncoding(X, rbF)
#         symbolization_XFs = RectangularBinEncoding(X, rbFs)

#         @test total_outcomes(x, symbolization_xN) == 5
#         @test_throws ArgumentError total_outcomes(x, symbolization_xF)
#         @test_throws ArgumentError total_outcomes(x, symbolization_xF)
#         @test_throws ArgumentError total_outcomes(x, symbolization_XNs)

#         @test total_outcomes(X, symbolization_XN) == 5^3
#         @test total_outcomes(X, symbolization_XNs) == 5*3*4
#         @test_throws ArgumentError total_outcomes(X, symbolization_XF)
#         @test_throws ArgumentError total_outcomes(X, symbolization_XFs)
#     end
# end
