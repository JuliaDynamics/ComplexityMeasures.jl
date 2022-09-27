@test CountOccurrences() isa CountOccurrences

D = Dataset(rand(1:3, 1000, 3))
ts = [(rand(1:4), rand(1:4), rand(1:4)) for i = 1:3000]
@test Entropies.entropy(Renyi(q = 2, base = 2), D, CountOccurrences()) isa Real
