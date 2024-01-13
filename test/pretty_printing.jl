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