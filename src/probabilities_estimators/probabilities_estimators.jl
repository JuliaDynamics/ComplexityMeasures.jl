function verify_counting_based(o)
    if !is_counting_based(o)
        s = "Outcome space $(o) isn't counting based."*
            "AddConstant requires a counting-based outcome space."
        throw(ArgumentError(s))
    end
end

include("RelativeAmount.jl")
include("Bayes.jl")
include("Shrinkage.jl")
include("AddConstant.jl")

# These don't actually result in probability distributions.
#include("SampleCoverage.jl")
#include("GoodTuring.jl")
