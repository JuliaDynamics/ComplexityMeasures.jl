using Entropies
using Test
using Random

@testset "Rectangular binning" begin

    x = Dataset(rand(Random.MersenneTwister(1234), 100_000, 2))
    push!(x, SVector(0, 0)) # ensure both 0 and 1 have values in, exactly.
    push!(x, SVector(1, 1))
    # All these binnings should give approximately same probabilities
    n = 10 # boxes cover 0 - 1 in steps of slightly more than 0.1
    ε = nextfloat(0.1, 2) # this guarantees that we get the same as the `n` above!

    binnings = [
        n,
        ε,
        RectangularBinning(n),
        RectangularBinning(ε),
        RectangularBinning([n, n]),
        RectangularBinning([ε, ε])
    ]

    for bin in binnings
        @testset "bin isa $(typeof(bin))" begin
            est = ValueHistogram(bin, x)
            p = probabilities(est, x)
            @test length(p) == 100
            @test all(e -> 0.009 ≤ e ≤ 0.011, p)

            p2, o = probabilities_and_outcomes(est, x)
            @test p2 == p
            @test o isa Vector{SVector{2, Float64}}
            @test length(o) == length(p)
            @test all(x -> x < 1, maximum(o))
            o2 = outcomes(est, x)
            @test o2 == o
        end
    end

    @testset "Check rogue 1s" begin
        b = RectangularBinning(0.1) # no `nextfloat` here, so the rogue (1, 1) is in extra bin!
        p = probabilities(ValueHistogram(b, x), x)
        @test length(p) == 100 + 1
        @test p[end] ≈ 1/100_000 atol = 1e-5
    end

    @testset "vector" begin
        x = rand(Random.MersenneTwister(1234), 100_000)
        push!(x, 0, 1)
        n = 10 # boxes cover 0 - 1 in steps of slightly more than 0.1
        ε = nextfloat(0.1) # this guarantees that we get the same as the `n` above!
        for bin in (RectangularBinning(n), RectangularBinning(ε))
            p = probabilities(ValueHistogram(bin, x), x)
            @test length(p) == 10
            @test all(e -> 0.09 ≤ e ≤ 0.11, p)
        end
    end

    # An extra bin might appear due to roundoff error after using nextfloat when
    # constructing `RectangularBinEncoding`s.
    # The following tests ensure with *some* certainty that this does not occur.
    @testset "Rogue extra bins" begin
        # Concrete examples where a rogue extra bin has appeared.
        x1 = [0.5213236385155418, 0.03516318860292644, 0.5437726723245310, 0.52598710966469610, 0.34199879802511246, 0.6017129426606275, 0.6972844365031351, 0.89163995617220900, 0.39009862510518045, 0.06296038912844315, 0.9897176284081909, 0.7935001082966890, 0.890198448900077700, 0.11762640519877565, 0.7849413168095061, 0.13768932585886573, 0.50869900547793430, 0.18042178201388548, 0.28507312391861270, 0.96480406570924970]
        x2 = [0.4125754262679051, 0.52844411982339560, 0.4535277505543355, 0.25502420827802674, 0.77862522996085940, 0.6081939026664078, 0.2628674795466387, 0.18846258495465185, 0.93320375283233840, 0.40093871561247874, 0.8032730760974603, 0.3531608285217499, 0.018436525139752136, 0.55541857934068420, 0.9907521337888632, 0.15382361136212420, 0.01774321666660561, 0.67569337507728300, 0.06130971689608822, 0.31417161558476836]
        N = 10
        b = RectangularBinning(N)
        rb1 = RectangularBinEncoding(b, x1; n_eps = 1)
        rb2 = RectangularBinEncoding(b, x1; n_eps = 2)
        @test encode(rb1, maximum(x1)) == -1 # shouldn't occur, but does when tolerance is too low
        @test encode(rb2, maximum(x1)) == 10

        rb1 = RectangularBinEncoding(b, x2; n_eps = 1)
        rb2 = RectangularBinEncoding(b, x2; n_eps = 2)
        @test encode(rb1, maximum(x2)) == -1 # shouldn't occur, but does when tolerance is too low
        @test encode(rb2, maximum(x2)) == 10
    end

end


@testset "Fixed Rectangular binning" begin

    x = Dataset(rand(Random.MersenneTwister(1234), 100_000, 2))
    push!(x, SVector(0, 0)) # ensure both 0 and 1 have values in, exactly.
    push!(x, SVector(1, 1))
    # All these binnings should give approximately same probabilities
    n = 10 # boxes cover 0 - 1 in steps of slightly more than 0.1
    ε = nextfloat(0.1, 2) # this guarantees that we get the same as the `n` above!

    bin = FixedRectangularBinning(0, 1, n, 2)

    est = ValueHistogram(bin)
    p = probabilities(est, x)
    @test length(p) == 100
    @test all(e -> 0.009 ≤ e ≤ 0.011, p)

    p2, o = probabilities_and_outcomes(est, x)
    @test p2 == p
    @test o isa Vector{SVector{2, Float64}}
    @test length(o) == length(p)
    @test all(x -> x < 1, maximum(o))
    o2 = outcomes(est, x)
    @test o2 == o

end
