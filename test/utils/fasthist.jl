using Entropies, Test

N = 100
t = collect(range(1, N; length = N))
D = Dataset(reshape(range(1, 3N; length = 3N), (N, 3)))

for x in (t, D)
    @test Entropies.fasthist!(x) isa Vector{Int}
    @test sum(Entropies.fasthist!(x)) == length(x) == N
    @test all(isequal(1), Entropies.fasthist!(x))
end