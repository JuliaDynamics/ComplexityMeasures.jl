using ComplexityMeasures, Test
using ComplexityMeasures: convert_logunit
# Constructors
@test LeonenkoProzantoSavani(Shannon()) isa LeonenkoProzantoSavani{<:Shannon}

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
ea = information(LeonenkoProzantoSavani(k = 5), rand(npts))
ea_n3 = information(LeonenkoProzantoSavani(Shannon(base = 3), k = 5), randn(npts))

@test U - max(0.01, U*0.03) ≤ ea ≤ U + max(0.01, U*0.03)
@test N_base3 * 0.98 ≤ ea_n3 ≤ N_base3 * 1.02



using Distributions: MvNormal
import Distributions.entropy as dentropy
function entropy(e::Renyi, 𝒩::MvNormal; base = 2)
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

# Eq. 15 in Nielsen & Nock (2011); https://arxiv.org/pdf/1105.3259.pdf
function entropy(e::Tsallis, 𝒩::MvNormal; base = 2)
    q = e.q
    Σ = 𝒩.Σ
    D = length(𝒩.μ)
    hr = entropy(Renyi(q = q), 𝒩; base)
    h = (exp((1 - q) * hr) - 1) / (1 - q)
    return convert_logunit(h, ℯ, base)
end
