using Entropies, Test

@testset "Spatial Permutation Entropy" begin
    x = [1 2 1; 8 3 4; 6 7 5]
    # Re-create the Ribeiro et al, 2012 using stencil
    # (you get 4 symbols in a 3x3 matrix. For this matrix, the upper left
    # and bottom right are the same symbol. So three probabilities in the end).
    stencil = CartesianIndex.([(1,0), (0,1), (1,1)])
    est = SpatialSymbolicPermutation(stencil, x, false)

    # Generic tests
    ps = probabilities(x, est)
    @test ps isa Probabilities
    @test length(ps) == 3
    @test sort(ps) == [0.25, 0.25, 0.5]

    h = entropy(Renyi(base = 2), x, est)
    @test h == 1.5

    # In fact, doesn't matter how we order the stencil,
    # the symbols will always be equal in top-left and bottom-right
    stencil = CartesianIndex.([(1,0), (1,1), (0,1)])
    est = SpatialSymbolicPermutation(stencil, x, false)
    @test entropy(Renyi(base = 2), x, est) == 1.5

    # But for sanity, let's ensure we get a different number
    # for a different stencil
    stencil = CartesianIndex.([(1,0), (2,0)])
    est = SpatialSymbolicPermutation(stencil, x, false)
    ps = sort(probabilities(x, est))
    @test ps[1] == 1/3
    @test ps[2] == 2/3

    # TODO: Symbolize tests once its part of the API.

    # Also test that it works for arbitrarily high-dimensional arrays
    stencil = CartesianIndex.([(0,1,0), (0,0,1), (1,0,0)])
    z = reshape(1:125, (5,5,5))
    est = SpatialSymbolicPermutation(stencil, z, false)
    # Analytically the total stencils are of length 4*4*4 = 64
    # but all of them given the same probabilities because of the layout
    ps = probabilities(z, est)
    @test ps == [1]
    # if we shuffle, we get random stuff
    using Random
    w = shuffle!(Random.MersenneTwister(42), collect(z))
    ps = probabilities(w, est)
    @test length(ps) > 1
end
