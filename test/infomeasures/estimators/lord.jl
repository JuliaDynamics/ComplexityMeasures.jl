using ComplexityMeasures, Test

# Constructors
@test Lord(Shannon()) isa Lord{<:Shannon}

# -------------------------------------------------------------------------------------
# Check if the estimator converge to true values for some distributions with
# analytically derivable entropy.
# -------------------------------------------------------------------------------------
# Entropy to log with base b of a uniform distribution on [0, 1] = ln(1 - 0)/(ln(b)) = 0
U = 0.00
# Entropy with natural log of 𝒩(0, 1) is 0.5*ln(2π) + 0.5.
N = round(0.5*log(2π) + 0.5, digits = 2)
N_base3 = ComplexityMeasures.convert_logunit(N, ℯ, 3)

npts = 20000 # a bit fewer points than for other tests, so tests don't take forever.
ea = information(Lord(k = 20), rand(npts))
ea_n3 = information(Lord(Shannon(base = 3), k = 20), randn(npts))

@test N_base3 * 0.96 ≤ ea_n3 ≤ N_base3 * 1.03
@test U - max(0.05, U*0.03) ≤ ea ≤ U + max(0.03, U*0.03)
