# Ensure unit conversion is correct
h_nats = 1.42
h_bits = convert_logunit(h_nats, ℯ, 2)
h_trits = convert_logunit(h_bits, 2, 3)
h_bans = convert_logunit(h_trits, 3, 10)
@test round(h_bits, digits = 3) ≈ 2.049
@test round(h_trits, digits = 3) ≈ 1.293
@test round(h_bans, digits = 3) ≈ 0.617
# A cycle of conversions returns to the same value.
@test h_nats ≈ convert_logunit(h_bans, 10, ℯ)

testfile("kozachenkoleonenko.jl")
testfile("kraskov.jl")
testfile("zhu.jl")
testfile("zhusingh.jl")
testfile("goria.jl")
testfile("alizadeharghami.jl")
testfile("correa.jl")
testfile("ebrahimi.jl")
testfile("vasicek.jl")
testfile("gao.jl")
testfile("lord.jl")
