function verify_counting_based(o, name = "BayesianRegularization")
    if !is_counting_based(o)
        s = "Outcome space $(o) isn't counting based."*
            "`$(name)` requires a counting-based outcome space."
        throw(ArgumentError(s))
    end
end

include("RelativeAmount.jl")
include("BayesianRegularization.jl")
include("Shrinkage.jl")
include("AddConstant.jl")
