using Test
using ComplexityMeasures
using ComplexityMeasures.DimensionalData: DimArray

# ----------------------------------------------------------------
# API
# ----------------------------------------------------------------
@test_throws MethodError Counts(rand(2)) # input must be integer array

x = rand(1:50, 5)
y = rand(1:50, 2, 2)
z = rand(1:50, 2, 2, 2)

# No labels provided: (`i`-th axis labels default to 1:size(x)[i] for i = 1:N, where `x`
# is the input array).
@test Counts(x) isa Counts{T, 1} where {T}
@test Counts(y) isa Counts{T, 2} where {T}
@test Counts(z) isa Counts{T, 3} where {T}

# Unnamed labels.
@test Counts(x, (1:5, )) isa Counts{T, 1} where {T}
@test Counts(y, (1:2, ['a', 'b'])) isa Counts{T, 2} where {T}
@test Counts(z, (1:2, ['a', 'b'], 7:8)) isa Counts{T, 3} where {T}

# Named labels.
@test Counts(x, (a = 1:5, )) isa Counts{T, 1} where {T}
@test Counts(y, (a = 1:2, b = ['a', 'b'])) isa Counts{T, 2} where {T}
@test Counts(z, (c = 1:2, d = ['a', 'b'], e = 7:8)) isa Counts{T, 3} where {T}

# Convenience
d = DimArray(x, (a = 1:5, ))
@test Counts(d) isa Counts{T, 1} where T

# ================================================================
# Outcomes
# ================================================================

# -----------------------------------------------------------------------------------------
# 1D data.
# -----------------------------------------------------------------------------------------
x = rand(1:50, 5)

# Unnamed dimensions.
outcomes(Counts(x, (1:5, ))) == 1:5
outcomes(Counts(x, (1:5, )), 1) == 1:5 # should be equivalent to not indexing
# Named dimensions.
outcomes(Counts(x, (a = 1:5, ))) == 1:5
outcomes(Counts(x, (a = 1:5, )), 1) == 1:5 # should be equivalent to not indexing

# -----------------------------------------------------------------------------------------
# 2D data.
# -----------------------------------------------------------------------------------------
y = rand(1:50, 2, 2)

# Unnamed dimensions.
outcomes(Counts(y, (1:2, ['a', 'b'],))) == (1:2, ['a', 'b'])
outcomes(Counts(y, (1:2, ['a', 'b'],)), 1:2) == (1:2, ['a', 'b'])
outcomes(Counts(y, (1:2, ['a', 'b'],)), 1) == 1:2
outcomes(Counts(y, (1:2, ['a', 'b'],)), 2) == ['a', 'b']

# Named dimensions.
outcomes(Counts(y, (a = 1:2, b = ['a', 'b'],))) == (1:2, ['a', 'b'])
outcomes(Counts(y, (a = 1:2, b = ['a', 'b'],)), 1:2) == (1:2, ['a', 'b'])
outcomes(Counts(y, (a = 1:2, b = ['a', 'b'],)), 1) == 1:2
outcomes(Counts(y, (a = 1:2, b = ['a', 'b'],)), 2) == ['a', 'b']

# -----------------------------------------------------------------------------------------
# 3D data.
# -----------------------------------------------------------------------------------------
z = rand(1:50, 2, 2, 2)

# Unnamed dimensions.
outs = (a = 1:2, b = ['a', 'b'], c = 7:8)
outs_unnamed = tuple(collect(o for o in outs)...) # outcomes doesn't care about dim names
@test outcomes(Counts(z, outs)) == outs_unnamed
@test all(outcomes(Counts(z, outs), i) == outs_unnamed[i] for i in eachindex(outs_unnamed))
@test outcomes(Counts(z, outs), 1:3) == outs_unnamed
@test outcomes(Counts(z, outs), 1:2) == outs_unnamed[1:2]

# Named dimensions.
outs = (a = 1:2, b = ['a', 'b'], c = 7:8)
outs_unnamed = tuple(collect(o for o in outs)...) # outcomes doesn't care about dim names
@test outcomes(Counts(z, outs)) == outs_unnamed
@test all(outcomes(Counts(z, outs), i) == outs_unnamed[i] for i in eachindex(outs_unnamed))
@test outcomes(Counts(z, outs), 1:3) == outs_unnamed
@test outcomes(Counts(z, outs), 1:2) == outs_unnamed[1:2]
