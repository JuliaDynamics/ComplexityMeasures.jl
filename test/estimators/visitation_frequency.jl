using Entropies
using Entropies.DelayEmbeddings, Test
using Random

@testset "Rectangular binning" begin
    x = Dataset(rand(Random.MersenneTwister(1234), 100_000, 2))
    push!(x, SVector(0, 0)) # ensure both 0 and 1 have values in, exactly.
    push!(x, SVector(1, 1))
    # All these binnings should give approximately same probabilities
    n = 10 # boxes cover 0 - 1 in steps of slightly more than 0.1
    ε = nextfloat(0.1) # this guarantees that we get the same as the `n` above!

    binnings = [
        RectangularBinning(n),
        RectangularBinning(ε),
        RectangularBinning([n, n]),
        RectangularBinning([ε, ε])
    ]

    for bin in binnings
        @testset "ϵ = $(bin.ϵ)" begin
            p = probabilities(x, bin)
            est = ValueHistogram(bin)
            p2 = probabilities(x, est)
            @test p == p2
            @test length(p) == 100
            @test all(e -> 0.009 ≤ e ≤ 0.011, p)
        end
    end

    @testset "Check rogue 1s" begin
        b = RectangularBinning(0.1) # no `nextfloat` here, so the rogue (1, 1) is in extra bin!
        p = probabilities(x, b)
        @test length(p) == 100 + 1
        @test p[end] ≈ 1/100_000 atol = 1e-5
    end

    @testset "vector" begin
        x = rand(Random.MersenneTwister(1234), 100_000)
        push!(x, 0, 1)
        n = 10 # boxes cover 0 - 1 in steps of slightly more than 0.1
        ε = nextfloat(0.1) # this guarantees that we get the same as the `n` above!
        binnings = RectangularBinning.((n, ε))
        for bin in binnings
            p = probabilities(x, bin)
            est = ValueHistogram(bin)
            p2 = probabilities(x, est)
            @test p == p2
            @test length(p) == 10
            @test all(e -> 0.09 ≤ e ≤ 0.11, p)
        end
    end
end
