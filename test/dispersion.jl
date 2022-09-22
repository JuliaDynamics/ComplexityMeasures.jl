using Entropies, Test

@testset "Dispersion methods" begin
    x = rand(100)

    @testset "Dispersion" begin

        @testset "Internals" begin
            # Li et al. (2018) recommends using at least 1000 data points when estimating
            # dispersion entropy.
            x = rand(1000)
            n_categories = 4
            m = 4
            τ = 1
            s = GaussianSymbolization(n_categories = n_categories)

            # Symbols should be in the set [1, 2, …, n_categories].
            symbols = Entropies.symbolize(x, s)
            @test all([s ∈ collect(1:n_categories) for s in symbols])

            # Dispersion patterns should have a normalized histogram that sums to 1.0.
            dispersion_patterns = DelayEmbeddings.embed(symbols, m, τ)
            hist = Entropies.dispersion_histogram(dispersion_patterns, length(x), m, τ)
            @test sum(hist) ≈ 1.0
        end

        # Test case from Rostaghi & Azami (2016)'s dispersion entropy paper. The
        # symbolization step is tested separately in the "Symbolization" test set.
        # We here start from pre-computed symbols `s`.
        s = [0.82,0.75,0.21,0.94,0.52,0.05,0.241,0.75,0.35,0.43,0.11,0.87]
        m = 2
        n_classes = 3
        scheme = GaussianSymbolization(n_classes)


        # Take only the non-zero probabilities from the paper (in `dispersion_histogram`,
        # we don't count zero-probability bins, so eliminate zeros for comparison).
        ps_paper = [1/11, 0/11, 3/11, 2/11, 1/11, 0/11, 1/11, 2/11, 1/11] |> sort
        ps_paper = ps_paper[findall(ps_paper .> 0)]

        ps = probabilities(x, Dispersion(m = m, s = scheme))
        @test ps isa Probabilities
        @test ps |> sort == ps_paper

        de_norm = entropy_renyi(x, Dispersion(normalize = true), q = 1, base = 2)
        @test 0.0 <= de_norm <= 1.0

        @test_throws ArgumentError entropy_renyi(x, Dispersion(normalize = true), q = 2)
    end
end
