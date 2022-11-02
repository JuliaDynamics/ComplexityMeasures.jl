using Test, StaticArrays, LinearAlgebra, Distributions

@testset "NN - Kraskov" begin
    m = 4
    τ = 1
    τs = tuple([τ*i for i = 0:m-1]...)
    x = rand(250)
    D = genembed(x, τs)
    est = Kraskov(k = 3, w = 1)
    @test entropy(est, D) isa Real
end

@testset "NN - KozachenkoLeonenko" begin
    m = 4
    τ = 1
    τs = tuple([τ*i for i = 0:m-1]...)
    x = rand(250)
    D = genembed(x, τs)
    est = KozachenkoLeonenko(w = 1)

    @test entropy(est, D) isa Real
end

@testset "NN - Zhu" begin

    # To ensure minimal rectangle volumes are correct, we also test internals directly here.
    # It's not feasible to construct an end-product test due to the neighbor searches.
    x = Dataset([[-1, -2], [0, -2], [3, 2]])
    y = Dataset([[3, 1], [-5, 1], [3, -2]])
    @test Entropies.volume_minirect([0, 0], x) == 24
    @test Entropies.volume_minirect([0, 0], y) == 40

    # Analytical tests for the estimated entropy
    DN = Dataset(randn(200000, 1))
    hN_base_e = 0.5 * log(MathConstants.e, 2π) + 0.5
    hN_base_2 = hN_base_e / log(2, MathConstants.e)

    e_base_e = Zhu(k = 3, base = MathConstants.e)
    e_base_2 = Zhu(k = 3, base = 2)

    @test round(entropy(e_base_e, DN), digits = 1) == round(hN_base_e, digits = 1)
    @test round(entropy(e_base_2, DN), digits = 1) == round(hN_base_2, digits = 1)
end


@testset "NN - ZhuSingh" begin

    using Test, Distributions, LinearAlgebra
    # Test internals in addition to end-product, because designing an exact end-product
    # test is  a mess due to the neighbor searches. If these top-level tests fail, then
    # the issue is probability related to these functions.
    using Test
    nns = Dataset([[-1, -1], [0, -2], [3, 2.0]])
    x = @SVector [0.0, 0.0]
    dists = Entropies.maxdists(x, nns)
    vol = Entropies.volume_rect(dists)
    ξ = Entropies.n_borderpoints(x, nns, dists)
    @test vol == 24.0
    @test ξ == 2.0

    nns = Dataset([[3, 1], [3, -2], [-5, 1.0]])
    x = @SVector [0.0, 0.0]
    dists = Entropies.maxdists(x, nns)
    vol = Entropies.volume_rect(dists)
    ξ = Entropies.n_borderpoints(x, nns, dists)
    @test vol == 40.0
    @test ξ == 2.0

    nns = Dataset([[-3, 1], [3, 1], [5, -2.0]])
    x = @SVector [0.0, 0.0]
    dists = Entropies.maxdists(x, nns)
    vol = Entropies.volume_rect(dists)
    ξ = Entropies.n_borderpoints(x, nns, dists)
    @test vol == 40.0
    @test ξ == 1.0

    # Analytical tests: 1D normal distribution
    DN = Dataset(randn(100000, 1))
    σ = 1.0
    hN_base_e = 0.5 * log(MathConstants.e, 2π * σ^2) + 0.5
    hN_base_2 = hN_base_e / log(2, MathConstants.e)

    e_base_e = ZhuSingh(k = 3, base = MathConstants.e)
    e_base_2 = ZhuSingh(k = 3, base = 2)

    @test round(entropy(e_base_e, DN), digits = 2) == round(hN_base_e, digits = 2)
    @test round(entropy(e_base_2, DN), digits = 2) == round(hN_base_2, digits = 2)

    # Analytical test: 3D normal distribution
    σs = ones(3)
    μs = zeros(3)
    𝒩₂ = MvNormal(μs, σs)
    Σ = diagm(σs)
    n = length(μs)
    h_𝒩₂_base_ℯ = 0.5n * log(ℯ, 2π) + 0.5*log(ℯ, det(Σ)) + 0.5n
    h_𝒩₂_base_2 = h_𝒩₂_base_ℯ  / log(2, ℯ)

    sample = Dataset(transpose(rand(𝒩₂, 50000)))
    hZS_𝒩₂_base_ℯ = entropy(e_base_e, sample)
    hZS_𝒩₂_base_2 = entropy(e_base_2, sample)

    # Estimation accuracy decreases for fixed N with increasing edimension, so exact comparison
    # isn't useful. Just check that values are within 1% of the target.
    tol_ℯ  = hZS_𝒩₂_base_ℯ * 0.01
    tol_2  = hZS_𝒩₂_base_2 * 0.01
    @test h_𝒩₂_base_ℯ - tol_ℯ ≤ hZS_𝒩₂_base_ℯ ≤ h_𝒩₂_base_ℯ + tol_ℯ
    @test h_𝒩₂_base_2 - tol_2 ≤ hZS_𝒩₂_base_2 ≤ h_𝒩₂_base_2 + tol_2
end
