@testset "Ordinal patterns" begin
    @test Entropies.encode_motif([2, 3, 1]) isa Int
    @test 0 <= Entropies.encode_motif([2, 3, 1]) <= factorial(3) - 1

    scheme = OrdinalPattern(m = 5, τ = 1)
    N = 100
    x = Dataset(repeat([1.1 2.2 3.3], N))
    y = Dataset(rand(N, 5))
    z = rand(N)

    # Without pre-allocation
    D = genembed(z, [0, -1, -2])
    scheme = OrdinalPattern(m = 5, τ = 2)

    @test Entropies.symbolize(z, scheme) isa Vector{<:Int}
    @test Entropies.symbolize(D, scheme) isa Vector{<:Int}


    # With pre-allocation
    N = 100
    x = rand(N)
    scheme = OrdinalPattern(m = 5, τ = 2)
    s = fill(-1, N-(scheme.m-1)*scheme.τ)

    # if symbolization has occurred, s must have been filled with integers in
    # the range 0:(m!-1)
    @test all(Entropies.symbolize!(s, x, scheme) .>= 0)
    @test all(0 .<= Entropies.symbolize!(s, x, scheme) .< factorial(scheme.m))

    m = 4
    D = Dataset(rand(N, m))
    s = fill(-1, length(D))
    @test all(0 .<= Entropies.symbolize!(s, D, scheme) .< factorial(m))
end

@testset "Gaussian symbolization" begin
     # Li et al. (2018) recommends using at least 1000 data points when estimating
    # dispersion entropy.
    x = rand(1000)
    c = 4
    m = 4
    τ = 1
    s = GaussianSymbolization(c = c)

    # Symbols should be in the set [1, 2, …, c].
    symbols = Entropies.symbolize(x, s)
    @test all([s ∈ collect(1:c) for s in symbols])

    # Test case from Rostaghi & Azami (2016)'s dispersion entropy paper.
    y = [9.0, 8.0, 1.0, 12.0, 5.0, -3.0, 1.5, 8.01, 2.99, 4.0, -1.0, 10.0]
    scheme = GaussianSymbolization(3)
    s = symbolize(y, scheme)
    @test s == [3, 3, 1, 3, 2, 1, 1, 3, 2, 2, 1, 3]
end

@testset "isless_rand" begin
    # because permutations are partially random, we sort many times and check that
    # we get *a* (not *the one*) correct answer every time
    for i = 1:50
        s = sortperm([1, 2, 3, 2], lt = Entropies.isless_rand)
        @test s == [1, 2, 4, 3] || s == [1, 4, 2, 3]
    end
end
