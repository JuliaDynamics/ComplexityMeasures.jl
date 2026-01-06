using ComplexityMeasures, Test
using Random: MersenneTwister
using DynamicalSystemsBase
using Integrals
cd(@__DIR__)

#test for all count-based outcome spaces

@testset begin "Count-based outcome spaces" 

    #simple 1d rand 
    x = rand(100)
    x_ue = rand([0.0:0.1:1.0;],100) #for unique elements

    #def count-based 1d outcome spaces 
    outcome_spaces = [BubbleSortSwaps(),AmplitudeAwareOrdinalPatterns(),OrdinalPatterns(),
        WeightedOrdinalPatterns(),CosineSimilarityBinning(),Dispersion(),
        SequentialPairDistances(x),UniqueElements(),ValueBinning(RectangularBinning(5))] 
    
    #build transferoperator from every outcomespace
    for ocs in outcome_spaces
        if ocs isa UniqueElements
            to = transferoperator(ocs,x_ue) #unique elements 
        else 
            to = transferoperator(ocs,x)
        end

        #test if transition matrix is normalized
        sum_rows = sum(to.transfermatrix;dims=2) 
        @test all( isapprox.(1.0, sum_rows ;atol = 1e-3))
    end

    #leave out spatial methods for now:
    #SpatialBubbleSortSwaps,SpatialDispersion,SpatialOrdinalPatterns
    #not trivial how to implement transferoperator!

end

#logistic map r=4 -> invariant measure is known analytically ρ(x) = 1/(π√x(1-x))

#stupidly simple example where we can actually compute the measure of
#each bin exactly. Calculate numerically integrated probabilities of bins for the analytical 
#invariant measure of the logistic map. Compare with probabilities, computed according to the 
#approximation to the transfer operator, not as naive visitation freqiencies to the bins.

@testset begin "Analytical logistic test probabilities" 

    ρ(x,p) = 1/(π*sqrt(x*(1-x))) #analytic invariant measure

    #-----------------logistic map orbit-------------------
    logistic_rule(x, p, n) = SVector{1}(p[1]*x[1]*(1.0 - x[1]))
    r = 4.0; p = [r]
    ds = DeterministicIteratedMap(logistic_rule,[0.4],p)
    orbit,t = trajectory(ds, 10^7; Ttr = 10^4)

    #--------------estimate invariant measure----------
    b = ValueBinning(FixedRectangularBinning(range(0,1;length=11),1,true))
    to = transferoperator(b,orbit)
    iv = invariantmeasure(to)
    p,outcomes = invariantmeasure(iv)
    @show outcomes

    #order from leftmost bin to rightmost bin  
    p_bins = p[sortperm(outcomes)]

    #----------compute probability for each bin analytically----------
    bin_ranges = to.outcome_space.binning.ranges[1]
    ρ_bins = zeros(10)

    for i in 1:length(bin_ranges)-1
        domain = (bin_ranges[i], bin_ranges[i+1]) # (lb, ub)
        prob = IntegralProblem(ρ, domain, p)
        sol = solve(prob, QuadGKJL())
        ρ_bins[i] = sol[1]
    end

    #--------------test if they're equal---------------
    #@show p_bins,ρ_bins
    @test all(isapprox.(p_bins,ρ_bins;atol=1e-3))

end

#test every kind of RectangularBinning with transferoperator

D = StateSpaceSet(rand(MersenneTwister(1234), 100, 2))

# Note that if `false` is used for `precise` the tests will fail.
# But that's okay, since we do not do guarantees for that case.
binnings = [
    RectangularBinning(3, true),
    RectangularBinning(0.2, true),
    RectangularBinning([2, 3], true),
    RectangularBinning([0.2, 0.3], true),
    FixedRectangularBinning(range(0, 1; length=5), 2, true)
]

@testset "Binning test $i" for i in eachindex(binnings)
    @show i
    b = binnings[i]
    to = transferoperator(ValueBinning(b),D)
    @test to isa ComplexityMeasures.TransferOperatorApproximation

    iv = invariantmeasure(to)
    @test iv isa InvariantMeasure

    p, bins = invariantmeasure(iv)
    @test p isa Probabilities
    @test bins isa Vector{Int}

    @test probabilities(TransferOperator(), ValueBinning(b) , D) isa Probabilities
    @test probabilities_and_outcomes(TransferOperator(), ValueBinning(b), D) isa Tuple{Probabilities, Vector{SVector{2, Float64}}}

    # Test that gives approximately same entropy as ValueBinning:
    abs(information(Shannon(), p) - information(ValueBinning(b), D) ) < 0.1 # or something like that
end

# Warn if we're not using precise binnings.
imprecise_warning = "`binning.precise == false`. You may be getting points outside the binning."
b = RectangularBinning(5)
@test_logs (:warn, imprecise_warning) transferoperator(ValueBinning(b), D; warn_precise = true)

#=
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
=#


#------------test API-------------

@testset begin "Test high-level invariantmeasure method"
    x = rand(1000)
    o = OrdinalPatterns{3}()

    #iterative method
    im_iter = invariantmeasure(o, x; method=:iterate)
    @test im_iter.to isa ComplexityMeasures.TransferOperatorApproximation
    ρ_iter = im_iter.ρ
    @test ρ_iter isa Probabilities

    #eigenvec method
    im_eigen = invariantmeasure(o, x; method=:eigen)
    @test im_eigen.to isa ComplexityMeasures.TransferOperatorApproximation
    ρ_eigen = im_eigen.ρ
    @test ρ_eigen isa Probabilities

    #compare if they're equal
    @test all(isapprox.(ρ_iter,ρ_eigen))

    #test through probabilities interface and compare 

    #try logistic map
    logistic_rule(x, p, n) = SVector{1}(p[1] * x[1] * (1.0 - x[1]))
    r = 4.0
    p = [r]
    ds = DeterministicIteratedMap(logistic_rule, [0.4], p)
    orbit, t = trajectory(ds, 10^7; Ttr=10^4)

    #try binning on logistic
    os = ValueBinning(RectangularBinning(10, true))
    p_TO = probabilities(TransferOperator(),os,orbit) #correct but and ordered as RelativeAmount
    p = probabilities(RelativeAmount(), os, orbit)
    @test all(isapprox.(p_TO.p, p.p; atol=1e-3))

    #try op on logistic
    op = OrdinalPatterns{3}()
    p_TO = probabilities(TransferOperator(), op, orbit) #correct but not ordered as RelativeAmount
    p = probabilities(RelativeAmount(), op, orbit)
    @show p_TO.p 
    @show p.p
    @test !all(isapprox.(p_TO.p, p.p; atol=1e-3))
end
