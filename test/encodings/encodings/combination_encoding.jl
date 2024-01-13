x = rand(3) # xᵢ ∈ [0, 1] for all xᵢ in x
e_ord = OrdinalPatternEncoding(length(x))
e_amp = RelativeMeanEncoding(0, 1, n = 3)
e_firstdiff = RelativeFirstDifferenceEncoding(0, 1, n = 2)
e_bin = RectangularBinEncoding(FixedRectangularBinning(0, 1, 2))
es = [e_ord, e_firstdiff, e_amp, e_bin]

# Constructors
# ----------------------------------------------------------------

@test CombinationEncoding(es) isa CombinationEncoding
@test CombinationEncoding(es...) isa CombinationEncoding

c = CombinationEncoding(OrdinalPatternEncoding())
@test_throws ArgumentError CombinationEncoding([c])

# Encoding/decoding
# ----------------------------------------------------------------
e_combo = CombinationEncoding(es)
symbol = encode(e_combo, x)
@test symbol isa Int
d = decode(e_combo, symbol)
@test d isa Tuple
@test length(d) == length(es)

# ----------------------------------------------------------------
# Analytical tests
# ----------------------------------------------------------------
encodings = [
    RelativeMeanEncoding(0, 1, n = 2),
    OrdinalPatternEncoding(3),
]
comboencoding = CombinationEncoding(encodings...)

x = [0.3, 0.1, 0.2]
s_a = encode(RelativeMeanEncoding(0, 1, n = 3), x) # mean = 0.2 => s_a = 1
s_o = encode(OrdinalPatternEncoding(3), x) # sorting pattern [2, 3, 1] => s_o = 4

# Symbol ranges are 1:2, 1:factorial(3) = 1:6, so we should get linear index 7
lidxs = LinearIndices((1:2, 1:factorial(3)))
s_c = encode(comboencoding, x)
@test lidxs[s_a, s_o] == 7 == s_c

@test decode(comboencoding, s_c)[1][1] ≈ 0.0 # left bin edge of first subinterval is zero
@test decode(comboencoding, s_c)[2] == [2, 3, 1] # idxs that would sort `x`

# ----------------------------------------------------------------
# Pretty printing
# ----------------------------------------------------------------
encodings = (
    RelativeFirstDifferenceEncoding(0, 1; n = 2),
    RelativeMeanEncoding(0, 1; n = 5),
    OrdinalPatternEncoding(3) # x is a three-element vector
    )
c = CombinationEncoding(encodings)
@test occursin("encodings = ", repr(c))
@test !occursin("linear_indices = ", repr(c))
@test !occursin("cartesian_indices = ", repr(c))