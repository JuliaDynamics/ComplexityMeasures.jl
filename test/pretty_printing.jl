using Test, ComplexityMeasures

# ----------------------------------------------------------------
# Types that doesn't require any special handling
# 
# Printing for types that require specialized printing is 
# tested in their respective source files.
# ----------------------------------------------------------------

# Example 1: only one field (so compact representation)
T = PlugIn
r = repr(T())
fns = fieldnames(T)
hidden_fields = ComplexityMeasures.hidefields(T)
displayed_fields = setdiff(fns, hidden_fields)
for fn in displayed_fields
    @test occursin("$fn = ", r)
end
for fn in hidden_fields
    @test !occursin("$fn = ", r)
end

# Example 2: more than one field (no expanded representation)
T = Renyi
r = repr(T())
fns = fieldnames(T)
hidden_fields = ComplexityMeasures.hidefields(T)
displayed_fields = setdiff(fns, hidden_fields)
for fn in displayed_fields
    @test occursin("$fn = ", r)
end
for fn in hidden_fields
    @test !occursin("$fn = ", r)
end

# A dummy information measure that contains all relevant types we want to test for.
# This is just to ensure complete test coverage for pretty printing in the case 
# of nested types.
struct DummyPrint <: InformationMeasure
    a
    b
    c
    d
    e
    f
end

m = DummyPrint(PlugIn(), Kraskov(), RelativeMeanEncoding(0, 1), OrdinalPatterns{3}(), 
    RelativeAmount(), MissingDispersionPatterns())
dummyestimator = PlugIn(m)
@test occursin("definition = ", repr(dummyestimator))



