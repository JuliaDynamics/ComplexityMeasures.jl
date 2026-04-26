using ComplexityMeasures, Test

# Constructors
@test Goria(Shannon()) isa Goria{<:Shannon}

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
ea_n = information(Goria(Shannon(base = ℯ), k = 5), randn(npts))
ea_n3 = information(Goria(Shannon(base = 3), k = 5), randn(npts))

@test N * 0.98 ≤ ea_n ≤ N * 1.02
@test N_base3 * 0.98 ≤ ea_n3 ≤ N_base3 * 1.02
