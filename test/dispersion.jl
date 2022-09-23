using Entropies, Test

@testset "Dispersion methods" begin
    x = rand(100)

    @testset "Dispersion" begin

        @testset "Internals" begin
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

            # Dispersion patterns should have a normalized histogram that sums to 1.0.
            dispersion_patterns = DelayEmbeddings.embed(symbols, m, τ)
            hist = Entropies.dispersion_histogram(dispersion_patterns, length(x), m, τ)
            @test sum(hist) ≈ 1.0
        end

        # Test case from Rostaghi & Azami (2016)'s dispersion entropy paper. The
        # symbolization step is tested separately in the "Symbolization" test set.
        # We here start from pre-computed symbols `s`.
        x = [0.82,0.75,0.21,0.94,0.52,0.05,0.241,0.75,0.35,0.43,0.11,0.87]
        m, n_classes = 2, 3
        est = Dispersion(m = m, s = GaussianSymbolization(n_classes), normalize = false)
        est_norm = Dispersion(m = m, s = GaussianSymbolization(n_classes), normalize = true)

        # Take only the non-zero probabilities from the paper (in `dispersion_histogram`,
        # we don't count zero-probability bins, so eliminate zeros for comparison).
        ps_paper = [1/11, 0/11, 3/11, 2/11, 1/11, 0/11, 1/11, 2/11, 1/11] |> sort
        ps_paper = ps_paper[findall(ps_paper .> 0)]

        ps = probabilities(x, est)
        @test ps |> sort == ps_paper

        # There is probably a typo in Rostaghi & Azami (2016). They state that the
        # non-normalized dispersion entropy is 1.8642. However, with identical probabilies,
        # we obtain non-normalized dispersion entropy of 1.8462.
        res = entropy_renyi(x, est, base = MathConstants.e, q = 1)
        @test round(res, digits = 4) == 1.8462

        # Again, probabilities are identical up to this point, but the values we get differ
        # slightly from the paper. They get normalized DE of 0.85, but we get 0.84. 0.85 is
        # the normalized DE you'd get by manually normalizing the (erroneous) value from
        # their previous step.
        res_norm = entropy_renyi(x, est_norm, base = MathConstants.e, q = 1)
        @test round(res_norm, digits = 2) == 0.84

        # Only defined for q = 1. There's potential in expanding the definition for q != 1,
        # but that requires careful thinking about the normalization step. A small future
        # paper on generalized dispersion entropy, perhaps?
        @test_throws ArgumentError entropy_renyi(x, Dispersion(normalize = true), q = 2)
    end

    @testset "Reverse dispersion entropy" begin
        # Reverse dispersion entropy is 0 when all probabilities are identical and equal
        # to 1/(n_classes^m).
        m, n_classes = 2, 2
        flat_dist = Probabilities(repeat([1/m^n_classes], m^n_classes))
        Hrde_minimal = Entropies.distance_to_whitenoise(flat_dist, n_classes, m,
            normalize = false)
        @test round(Hrde_minimal, digits = 7) ≈ 0.0

        # Reverse dispersion entropy is maximal when there is only one non-zero dispersal
        # pattern. Then reverse dispersion entropy is
        # 1 - 1/(n_classes^m). When normalizing to this value, the RDE should be 1.0.
        m, n_classes = 2, 2
        single_element_dist = Probabilities([1.0, 0.0, 0.0, 0.0])
        Hrde_maximal = Entropies.distance_to_whitenoise(single_element_dist, n_classes, m,
            normalize = false)
        Hrde_maximal_norm = Entropies.distance_to_whitenoise(single_element_dist, n_classes, m,
            normalize = true)
        @test round(Hrde_maximal, digits = 7) ≈ 1 - 1/(n_classes^m)
        @test round(Hrde_maximal_norm, digits = 7) ≈ 1.0
    end
end
