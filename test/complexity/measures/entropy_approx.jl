using ComplexityMeasures
using DynamicalSystemsBase
using Statistics

@test_throws UndefKeywordError ApproximateEntropy()
@test_throws ArgumentError complexity(ApproximateEntropy(r = 0.2), StateSpaceSet(rand(100, 3)))

# ApproximateEntropy is not currently normalizable
x = rand(100)
@test_throws ArgumentError complexity_normalized(ApproximateEntropy(r = 0.05, m = 2), x)

# Here, we try to reproduce Pincus' results within reasonable accuracy
# for the Henon map. He doesn't give initial conditions, so we just check that our
#  results +- 1σ approaches what he gets for this system for time series length 1000).
# ---------------------------------------------------------------------------
# Equation 13 in Pincus (1991)
function eom_henon(u, p, n)
    R = p[1]
    x, y = u
    dx = R*y + 1 - 1.4*x^2
    dy = 0.3*R*x
    return SVector(dx, dy)
end
henon(; u₀ = rand(2), R = 0.8) = DeterministicIteratedMap(eom_henon, u₀, [R])

# For some initial conditions, the Henon map as specified here blows up,
# so we need to check for infinite values.
containsinf(x) = any(isinf.(x))

function calculate_hs(; nreps = 50, L = 1000)
    # Calculate approx entropy for 50 different initial conditions
    hs = zeros(nreps)
    hs_conv = zeros(nreps)
    k = 1
    while k <= nreps
        sys = henon(u₀ = rand(2), R = 0.8)
        t, tvec = trajectory(sys, L, Ttr = 5000)

        if !any(tᵢ -> containsinf(tᵢ), t)
            x = t[:, 1]
            hs[k] = complexity(ApproximateEntropy(r = 0.05, m = 2), x)
            hs_conv[k] = entropy_approx(x, r = 0.05, m = 2)
            k += 1
        end
    end
    return hs, hs_conv
end
hs, hs_conv = calculate_hs()

@test mean(hs) - std(hs) <= 0.385 <= mean(hs) + std(hs)
@test mean(hs_conv) - std(hs_conv) <= 0.385 <= mean(hs_conv) + std(hs_conv)
