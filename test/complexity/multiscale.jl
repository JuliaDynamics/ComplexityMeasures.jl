using Entropies

x = rand(1000)
c = ReverseDispersion()
maxscale = 5
mcr = Entropies.multiscale(Regular(), c, x; maxscale)
mcc = Entropies.multiscale(Composite(), c, x; maxscale)
mcrn = Entropies.multiscale_normalized(Regular(), c, x; maxscale)
mccn = Entropies.multiscale_normalized(Composite(), c, x; maxscale)
@test mcr isa Vector{T} where T <: Real
@test mcc isa Vector{T} where T <: Real
@test mcrn isa Vector{T} where T <: Real
@test mccn isa Vector{T} where T <: Real
@test length(mcr) == 5
@test length(mcc) == 5
@test length(mcrn) == 5
@test length(mccn) == 5
