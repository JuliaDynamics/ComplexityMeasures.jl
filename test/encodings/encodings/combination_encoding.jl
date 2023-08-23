x = rand(3) # xᵢ ∈ [0, 1] for all xᵢ in x
e_ord = OrdinalPatternEncoding(length(x))
e_amp = RelativeMeanEncoding(0, 1, n = 3)
e_firstdiff = RelativeFirstDifferenceEncoding(0, 1, n = 2)
e_bin = RectangularBinEncoding(FixedRectangularBinning(0, 1, 2))
es = [e_ord, e_firstdiff, e_amp, e_bin]

e_combo = CombinationEncoding(es)

symbol = encode(e_combo, x)
@test symbol isa Int
d = decode(e_combo, symbol)
@test d isa AbstractVector
@test length(d) == length(es)

c = CombinationEncoding(OrdinalPatternEncoding())
@test_throws ArgumentError CombinationEncoding([c])
