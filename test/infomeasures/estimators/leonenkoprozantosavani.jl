using ComplexityMeasures, Test
using ComplexityMeasures: convert_logunit
import ComplexityMeasures: information
using Random
rng = Xoshiro(1234)

# Constructors
@test LeonenkoProzantoSavani(Shannon()) isa LeonenkoProzantoSavani{<:Shannon}

@test_throws ArgumentError LeonenkoProzantoSavani(Kaniadakis())
@test_throws ArgumentError LeonenkoProzantoSavani(k = 1)

# -------------------------------------------------------------------------------------
# Check if the estimator converge to true values for some distributions with
# analytically derivable entropy.
# -------------------------------------------------------------------------------------
# Entropy to log with base b of a uniform distribution on [0, 1] = ln(1 - 0)/(ln(b)) = 0
U = 0.00
# Entropy with natural log of 𝒩(0, 1) is 0.5*ln(2π) + 0.5.
N = round(0.5*log(2π) + 0.5, digits = 2)
N_base3 = ComplexityMeasures.convert_logunit(N, ℯ, 3)

npts = 100000
ea = information(LeonenkoProzantoSavani(k = 5), rand(rng, npts))
ea_n3 = information(LeonenkoProzantoSavani(Shannon(base = 3), k = 5), randn(rng, npts))

@test U - max(0.01, U*0.03) ≤ ea ≤ U + max(0.01, U*0.03)
@test N_base3 * 0.98 ≤ ea_n3 ≤ N_base3 * 1.02

# -------------------------------------------------------------------------------------
# Renyi entropy.
# ------------------------------------------------------------------------------------
using Distributions: MvNormal
import Distributions.entropy as dentropy
function information(e::Renyi, 𝒩::MvNormal; base = 2)
    q = e.q
    if q ≈ 1.0
        h = dentropy(𝒩)
    else
        Σ = 𝒩.Σ
        D = length(𝒩.μ)
        h = dentropy(𝒩) - (D / 2) * (1 + log(q) / (1 - q))
    end
    return convert_logunit(h, ℯ, base)
end

# We know the analytical expression for the Rényi entropy of a multivariate normal.
# It is implemented in the function above.
𝒩 = MvNormal([0, 1], [1, 1])
h_true = information(Renyi(q = 2), 𝒩, base = 2)
𝒩pts = StateSpaceSet(transpose(rand(rng, 𝒩, npts)))
h_estimated = information(LeonenkoProzantoSavani(Renyi(q = 2, base = 2), k = 10), 𝒩pts)

# Check that we're less than 10% off target
@test abs(h_estimated - h_true)/h_true < 0.1

# -------------------------------------------------------------------------------------
# Tsallis entropy.
# ------------------------------------------------------------------------------------
# Eq. 15 in Nielsen & Nock (2011); https://arxiv.org/pdf/1105.3259.pdf
function information(e::Tsallis, 𝒩::MvNormal; base = 2)
    q = e.q
    Σ = 𝒩.Σ
    D = length(𝒩.μ)
    hr = information(Renyi(q = q), 𝒩; base)
    h = (exp((1 - q) * hr) - 1) / (1 - q)
    return convert_logunit(h, ℯ, base)
end

# We know the analytical expression for the Tsallis entropy of a multivariate normal.
# It is implemented in the function above.
𝒩 = MvNormal([0, 1], [1, 1])
h_true = information(Tsallis(q = 2), 𝒩, base = 2)
𝒩pts = StateSpaceSet(transpose(rand(rng, 𝒩, npts)))
h_estimated = information(LeonenkoProzantoSavani(Tsallis(q = 2, base = 2), k = 10), 𝒩pts)

# Check that we're less than 10% off target
@test abs(h_estimated - h_true)/h_true < 0.1
