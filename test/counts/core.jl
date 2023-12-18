using Test
using ComplexityMeasures

# ----------------------------------------------------------------
# API
# ----------------------------------------------------------------
@test_throws MethodError Counts(rand(2)) # input must be integer array

x = rand(1:50, 5)
y = rand(1:50, 2, 2)
z = rand(1:50, 2, 2, 3)

# the outcomes
ox = (collect(1:5),)
oy = (collect(1:2), collect(1:2),)
oz = (collect(1:2), collect(1:2), collect(1:2))

# No labels provided: (`i`-th axis labels default to 1:size(x)[i] for i = 1:N, where `x`
# is the input array).
@test Counts(x, ox) isa Counts{T, 1} where {T}
@test Counts(y, oy) isa Counts{T, 2} where {T}
@test Counts(z, oz) isa Counts{T, 3} where {T}

# Unnamed labels (should work for different outcome types)
@test Counts(x, (1:5, )) isa Counts{T, 1} where {T}
@test Counts(y, (1:2 |> collect, ['a', 'b'])) isa Counts{T, 2} where {T}
@test Counts(z, (1:2, ['a', 'b'], 7:8)) isa Counts{T, 3} where {T}


# ================================================================
# Outcomes
# ================================================================

# -----------------------------------------------------------------------------------------
# 1D data.
# -----------------------------------------------------------------------------------------
x = rand(1:50, 5)

# Unnamed dimensions.
@test outcomes(Counts(x)) == Outcome(1):1:Outcome(5)
@test outcomes(Counts(x, (1:5, ))) == 1:5
@test outcomes(Counts(x, (1:5, )), 1) == 1:5 # should be equivalent to not indexing

# The number of counts and outcomes must match.
@test_throws ArgumentError Counts(rand(1:3, 10), (1:9,))

# -----------------------------------------------------------------------------------------
# 2D data.
# -----------------------------------------------------------------------------------------
y = rand(1:50, 2, 2)

# Unnamed dimensions.
@test outcomes(Counts(y, (1:2, ['a', 'b'],))) == (1:2, ['a', 'b'])
@test outcomes(Counts(y, (1:2, ['a', 'b'],)), 1:2) == (1:2, ['a', 'b'])
@test outcomes(Counts(y, (1:2, ['a', 'b'],)), 1) == 1:2
@test outcomes(Counts(y, (1:2, ['a', 'b'],)), 2) == ['a', 'b']

# -----------------------------------------------------------------------------------------
# 3D data.
# -----------------------------------------------------------------------------------------
z = rand(1:50, 2, 2, 2)

# Unnamed dimensions.
outs = (1:2, ['a', 'b'], 7:8)
outs_unnamed = tuple(collect(o for o in outs)...) # outcomes doesn't care about dim names
@test outcomes(Counts(z, outs)) == outs_unnamed
@test all(outcomes(Counts(z, outs), i) == outs_unnamed[i] for i in eachindex(outs_unnamed))
@test outcomes(Counts(z, outs), 1:3) == outs_unnamed
@test outcomes(Counts(z, outs), 1:2) == outs_unnamed[1:2]

