using ComplexityMeasures
using Test
using Random
rng = Xoshiro(1234)

# ----------------------------------------------------------------
# lorenz_curve
# ----------------------------------------------------------------
p = [0.2, 0.5, 0.3]
@test lorenz_curve(p) ≈ [0.5, 0.8, 1.0]
@test lorenz_curve(Probabilities(p)) ≈ [0.5, 0.8, 1.0]
@test lorenz_curve([1.0]) == [1.0]
# zeros are handled by sorting
@test lorenz_curve([0.0, 1.0, 0.0]) == [1.0, 1.0, 1.0]
# the curve is invariant under permutations
@test lorenz_curve(p) == lorenz_curve(p[[3, 1, 2]])

# ----------------------------------------------------------------
# majorizes
# ----------------------------------------------------------------
N = 5
deterministic = Probabilities([1.0; zeros(N - 1)]; normed = true)
uniform = Probabilities(fill(1 / N, N))
random = Probabilities(rand(rng, N))

# extremal elements of the order
@test majorizes(deterministic, uniform)
@test majorizes(deterministic, random)
@test majorizes(random, uniform)
@test !majorizes(uniform, deterministic)
@test !majorizes(uniform, random)

# reflexivity and invariance under permutations
@test majorizes(random, random)
perm = randperm(rng, N)
@test majorizes(random, random[perm])
@test majorizes(random[perm], random)

# a comparable pair: p ≻ q but not q ≻ p
p = [0.5, 0.3, 0.2]
q = [0.4, 0.35, 0.25]
@test majorizes(p, q)
@test !majorizes(q, p)

# an incomparable pair: first partial sum favours one, second the other
a = [0.6, 0.2, 0.2]
b = [0.5, 0.5, 0.0]
@test !majorizes(a, b)
@test !majorizes(b, a)

# tolerance
@test majorizes(b, a; atol = 0.1)
@test !majorizes(b, a; atol = 0.05)

# input checks
@test_throws ArgumentError majorizes(a, [0.5, 0.5])
@test_throws ArgumentError majorizes([2.0, 0.0], [1.0, 1.0])
@test majorizes([2.0, 0.0], [1.0, 1.0]; checknorm = false)
@test !majorizes([1.0, 1.0], [2.0, 0.0]; checknorm = false)

# ----------------------------------------------------------------
# Schur-concavity: p ≻ q implies H(p) ≤ H(q) for the discrete measures
# ----------------------------------------------------------------
measures = [
    Shannon(),
    Renyi(; q = 0.5), Renyi(; q = 2.0),
    Tsallis(; q = 0.5), Tsallis(; q = 2.0),
    Curado(),
    StretchedExponential(; base = ℯ),
    Kaniadakis(),
    Identification(),
    ShannonExtropy(),
    RenyiExtropy(; q = 2.0),
    TsallisExtropy(; q = 2.0),
]

# Mixing with the uniform distribution, or averaging over permutations, is a doubly
# stochastic map, so the result is always majorized by the input.
function majorized_by(rng, p)
    t = rand(rng)
    u = fill(1 / length(p), length(p))
    perm = randperm(rng, length(p))
    return Probabilities((1 - t) .* (p .+ p[perm]) ./ 2 .+ t .* u)
end

@testset "Schur-concavity of $(nameof(typeof(m)))" for m in measures
    for (p, q) in [(deterministic, random), (random, uniform),
            (Probabilities([0.5, 0.3, 0.2]), Probabilities([0.4, 0.35, 0.25]))]
        @test majorizes(p, q)
        @test information(m, p) ≤ information(m, q)
    end
    for _ in 1:50
        L = rand(rng, 2:8)
        p = Probabilities(rand(rng, L))
        q = majorized_by(rng, p)
        @test majorizes(p, q; atol = 1e-12)
        @test information(m, p) ≤ information(m, q) + 1e-12
    end
end
