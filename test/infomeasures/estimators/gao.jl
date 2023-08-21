using ComplexityMeasures, Test

# Constructors
@test Gao(Shannon()) isa Gao{<:Shannon}

# -------------------------------------------------------------------------------------
# Check if the estimator converge to true values for some distributions with
# analytically derivable entropy.
# -------------------------------------------------------------------------------------
# Entropy to log with base b of a uniform distribution on [0, 1] = ln(1 - 0)/(ln(b)) = 0
U = 0.00
# Entropy with natural log of 𝒩(0, 1) is 0.5*ln(2π) + 0.5.
N = round(0.5*log(2π) + 0.5, digits = 2)
N_base3 = ComplexityMeasures.convert_logunit(N, ℯ, 3)

# Without correction
# ------------------------------------------------------------------------------------
npts = 1000000
ea = information(Gao(k = 5, corrected = false), rand(npts))
ea_n = information(Gao(Shannon(base = ℯ), k = 5, corrected = false), randn(npts))
ea_n3 = information(Gao(Shannon(base = 3), k = 5, corrected = false), randn(npts))

# It is not expected that this estimator will be precise, so increase
# allowed error bounds compared to other estimators.
@test U - max(0.2, U*0.2) ≤ ea ≤ U + max(0.1, U*0.2)
@test N * 0.8 ≤ ea_n ≤ N * 1.02
@test N_base3 * 0.8 ≤ ea_n3 ≤ N_base3 * 1.02


# With correction
# ------------------------------------------------------------------------------------
ea = information(Gao(k = 5, corrected = true), rand(npts))
ea_n = information(Gao(Shannon(base = ℯ), k = 5, corrected = true), randn(npts))

@test U - max(0.01, U*0.03) ≤ ea ≤ U + max(0.01, U*0.03)
@test N * 0.98 ≤ ea_n ≤ N * 1.02
