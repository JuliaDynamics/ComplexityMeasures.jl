using Revise 
using ComplexityMeasures, Test
using Random: MersenneTwister
using DynamicalSystemsBase
using Integrals
@test TransferOperator(RectangularBinning(3)) isa TransferOperator

D = StateSpaceSet(rand(MersenneTwister(1234), 100, 2))

# Note that if `false` is used for `precise` the tests will fail.
# But that's okay, since we do not do guarantees for that case.
binnings = [
    RectangularBinning(3, true),
    RectangularBinning(0.2, true),
    RectangularBinning([2, 3], true),
    RectangularBinning([0.2, 0.3], true),
    FixedRectangularBinning(range(0, 1; length = 5), 2, true)
]

# There's not easy way of constructing an analytical example for the resulting
# probabilities, because they are computed according to the approximation to the
# transfer operator, not as naive visitation freqiencies to the bins. We just test
# generically here.

# TODO: make a stupidly simple example where we can actually compute the measure of
# each bin exactly.

#logistic map r=4 -> invariant measure is known analytically ρ(x) = 1/(π√x(1-x))

@testset begin "Analytical logistic test probabilities" 

    ρ(x,p) = 1/(π*sqrt(x*(1-x))) #analytic invariant measure

    #-----------------logistic map orbit-------------------
    logistic_rule(x, p, n) = SVector{1}(p[1]*x[1]*(1.0 - x[1]))
    r = 4.0; p = [r]
    ds = DeterministicIteratedMap(logistic_rule,[0.4],p)
    orbit,t = trajectory(ds, 10^7; Ttr = 10^4)

    #--------------estimate invariant measure----------
    b = RectangularBinning(10, true)
    to = ComplexityMeasures.transferoperator(orbit, b)
    iv = invariantmeasure(to)
    p,outcomes = invariantmeasure(iv)

    #order from leftmost bin to rightmost bin  
    p_bins = p[sortperm(outcomes)]

    #----------compute probability for each bin analytically----------
    bin_ranges = to.encoding.ranges[1]
    ρ_bins = zeros(10)

    for i in 1:length(bin_ranges)-1
        domain = (bin_ranges[i], bin_ranges[i+1]) # (lb, ub)
        prob = IntegralProblem(ρ, domain, p)
        sol = solve(prob, QuadGKJL())
        ρ_bins[i] = sol[1]
    end

    #--------------test if they're equal---------------
    @show p_bins,ρ_bins
    @test all(isapprox.(p_bins,ρ_bins;atol=1e-3))

end

@testset "Binning test $i" for i in eachindex(binnings)
    b = binnings[i]
    to = ComplexityMeasures.transferoperator(D, b)
    @test to isa ComplexityMeasures.TransferOperatorApproximationRectangular

    iv = invariantmeasure(to)
    @test iv isa InvariantMeasure

    p, bins = invariantmeasure(iv)
    @test p isa Probabilities
    @test bins isa Vector{Int}

    o = TransferOperator(binnings[i])
    @test probabilities(o, D) isa Probabilities
    @test probabilities_and_outcomes(o, D) isa Tuple{Probabilities, Vector{SVector{2, Float64}}}

    # Test that gives approximately same entropy as ValueBinning:
    abs(information(TransferOperator(b), D) - information(ValueBinning(b), D) ) < 0.1 # or something like that
end

# Warn if we're not using precise binnings.
imprecise_warning = "`binning.precise == false`. You may be getting points outside the binning."
b = RectangularBinning(5)
@test_logs (:warn, imprecise_warning) ComplexityMeasures.transferoperator(D, b; warn_precise = true)

# ---------------
# Reproducibility
# ---------------

# Resetting rng will lead to identical results.
p1 = probabilities(TransferOperator(b; rng = MersenneTwister(1234)), D)
p2 = probabilities(TransferOperator(b; rng = MersenneTwister(1234)), D)
@test all(p1 .== p2)

# Not resetting rng will lead to non-identical results.
rng = MersenneTwister(1234)
p1 = probabilities(TransferOperator(b; rng), D)
p2 = probabilities(TransferOperator(b; rng), D)
@test !all(p1 .== p2)
@test p1[1] ≈ p2[1] # But we should be close
