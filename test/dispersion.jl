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

        ps = probabilities(x, Dispersion())
        @test ps isa Probabilities

        de = entropy_renyi(x, Dispersion(normalize = false), q = 1, base = 2)
        de_norm = entropy_renyi(x, Dispersion(normalize = true), q = 1, base = 2)
        @test de isa Real
        @test de_norm isa Real
        @test de >= 0.0
        @test de_norm >= 0.0
    end

    @testset "ReverseDispersion" begin
        est = ReverseDispersion()
        ps = probabilities(x, est)
        @test ps isa Probabilities

        de = entropy_renyi(x, ReverseDispersion(normalize = false), q = 1, base = 2)
        de_norm = entropy_renyi(x, ReverseDispersion(normalize = true), q = 1, base = 2)
        @test de isa Real
        @test de_norm isa Real
        @test de >= 0.0
        @test de_norm >= 0.0
    end

end
