using Test, ComplexityMeasures
using Random; rng = MersenneTwister(1234)
using DelayEmbeddings

x = rand(10000)
m = 5
x_embed = embed(x, m, 1)
encoding = BubbleSortSwapsEncoding{m}()
symbols = encode.(Ref(encoding), x_embed.data)
@test all(0 .<= symbols .<= (m * (m - 1)) ÷ 2)