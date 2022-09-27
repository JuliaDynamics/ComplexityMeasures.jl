x = rand(100)
@test reverse_dispersion(x) isa Real
@test 0.0 <= reverse_dispersion(x, normalize = true) <= 1.0

@testset "Distance to whitenoise" begin
    m, n_classes = 2, 2
    est = Dispersion(m = m, symbolization = GaussianSymbolization(c = n_classes))

        # Reverse dispersion entropy is 0 when all probabilities are identical and equal
    # to 1/(n_classes^m).
    flat_dist = Probabilities(repeat([1/m^n_classes], m^n_classes))
    Hrde_minimal = distance_to_whitenoise(flat_dist, est, normalize = false)
    @test round(Hrde_minimal, digits = 7) ≈ 0.0

        # Reverse dispersion entropy is maximal when there is only one non-zero dispersal
    # pattern. Then reverse dispersion entropy is
    # 1 - 1/(n_classes^m). When normalizing to this value, the RDE should be 1.0.
    m, n_classes = 2, 2
    single_element_dist = Probabilities([1.0, 0.0, 0.0, 0.0])
    Hrde_maximal = distance_to_whitenoise(single_element_dist, est,
        normalize = false)
    Hrde_maximal_norm = distance_to_whitenoise(single_element_dist, est,
        normalize = true)
    @test round(Hrde_maximal, digits = 7) ≈ 1 - 1/(n_classes^m)
    @test round(Hrde_maximal_norm, digits = 7) ≈ 1.0
end
