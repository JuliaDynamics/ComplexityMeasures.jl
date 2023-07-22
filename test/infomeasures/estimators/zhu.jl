using ComplexityMeasures, Test

# Constructors
@test Zhu(Shannon()) isa Zhu{<:Shannon}

# To ensure minimal rectangle volumes are correct, we also test internals directly here.
# It's not feasible to construct an end-product test due to the neighbor searches.
x = StateSpaceSet([[-1, -2], [0, -2], [3, 2]]);
y = StateSpaceSet([[3, 1], [-5, 1], [3, -2]]);
@test ComplexityMeasures.volume_minimal_rect([0, 0], x) == 24
@test ComplexityMeasures.volume_minimal_rect([0, 0], y) == 40

# -------------------------------------------------------------------------------------
# Check if the estimator converge to true values for some distributions with
# analytically derivable entropy.
# -------------------------------------------------------------------------------------
# Entropy to log with base b of a uniform distribution on [0, 1] = ln(1 - 0)/(ln(b)) = 0
U = 0.00
# Entropy with natural log of 𝒩(0, 1) is 0.5*ln(2π) + 0.5.
N = round(0.5*log(2π) + 0.5, digits = 2)
N_base3 = ComplexityMeasures.convert_logunit(N, ℯ, 3)

npts = 1000000
ea = information(Zhu(k = 5), rand(npts))
ea_n3 = information(Zhu(k = 5, base = 3), randn(npts))

@test U - max(0.01, U*0.03) ≤ ea ≤ U + max(0.01, U*0.03)
@test N_base3 * 0.98 ≤ ea_n3 ≤ N_base3 * 1.02
