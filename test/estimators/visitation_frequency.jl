@test VisitationFrequency(RectangularBinning(3)) isa VisitationFrequency

D = Dataset(rand(100, 3))

@testset "Counting visits" begin
    @test Entropies.marginal_visits(D, RectangularBinning(0.2), 1:2) isa Vector{<:AbstractVector{Int}}
    @test Entropies.joint_visits(D, RectangularBinning(0.2)) isa Vector{<:AbstractVector{Int}}
end

binnings = [
    RectangularBinning(3),
    RectangularBinning(0.2),
    RectangularBinning([2, 2, 3]),
    RectangularBinning([0.2, 0.3, 0.3])
]

@testset "Binning test $i" for i in eachindex(binnings)
    est = VisitationFrequency(binnings[i])
    @test probabilities(D, est) isa Probabilities
    @test entropy(Renyi(q = 1, base = 3), D, est) isa Real # Regular order-1 entropy
    @test entropy(Renyi(q = 3, base = 2), D, est) isa Real # Higher-order entropy
    @test entropy(Renyi(q = 3, base = 1), D, est) isa Real # Higher-order entropy

end
