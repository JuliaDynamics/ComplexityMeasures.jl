using ComplexityMeasures, Test

# Constructors
@test Ebrahimi(Shannon()) isa Ebrahimi{<:Shannon}

# -------------------------------------------------------------------------------------
# Check if the estimator converge to true values for some distributions with
# analytically derivable entropy.
# -------------------------------------------------------------------------------------
# Entropy to log with base b of a uniform distribution on [0, 1] = ln(1 - 0)/(ln(b)) = 0
U = 0.0
# Entropy with natural log of 𝒩(0, 1) is 0.5*ln(2π) + 0.5.
N = round(0.5 * log(2π) + 0.5, digits = 2)
N_base3 = ComplexityMeasures.convert_logunit(N, ℯ, 3)

npts = 1000000
ea = information(Ebrahimi(m = 100), rand(npts))
ea_n = information(Ebrahimi(Shannon(base = ℯ), m = 100), randn(npts))
ea_n3 = information(Ebrahimi(Shannon(base = 3), m = 100), randn(npts))

@test U - max(0.02, U * 0.03) ≤ ea ≤ U + max(0.01, U * 0.03)
@test N * 0.96 ≤ ea_n ≤ N * 1.02
@test N_base3 * 0.96 ≤ ea_n3 ≤ N_base3 * 1.02
