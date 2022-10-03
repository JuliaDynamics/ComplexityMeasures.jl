using DynamicalSystemsBase
using Statistics

@test_throws ArgumentError approx_entropy(Dataset(rand(100, 3)))

# Here, we try to reproduce Pincus' results within reasonable accuracy
# for the Henon map. He doesn't give initial conditions, so we just check that our
#  results +- 1σ approaches what he gets for this system for time series length 1000).
# ---------------------------------------------------------------------------
# Equation 13 in Pincus (1991)
function eom_henon(u, p, n)
    R = p[1]
    x, y = (u...,)
    dx = R*y + 1 - 1.4*x^2
    dy = 0.3*R*x

    return SVector{2}(dx, dy)
end
henon(; u₀ = rand(2), R = 0.8) = DiscreteDynamicalSystem(eom_henon, u₀, [R])

# For some initial conditions, the Henon map as specified here blows up,
# so we need to check for infinite values.
containsinf(x) = any(isinf.(x))

function calculate_hs(; nreps = 50, L = 1000)
    # Calculate approx entropy for 50 different initial conditions
    hs = zeros(nreps)
    k = 1
    while k <= nreps
        sys = henon(u₀ = rand(2), R = 0.8)
        t = trajectory(sys, L, Ttr = 5000)

        if !any([containsinf(tᵢ) for tᵢ in t])
            x = t[:, 1]
            hs[k] = approx_entropy(x, r = 0.05, m = 2)
            k += 1
        end
    end
    return hs
end
hs = calculate_hs()
@test mean(hs) - std(hs) <= 0.385 <= mean(hs) + std(hs)
