export JackknifeEstimator

struct JackknifeEstimator{I <: InformationMeasure} <: DiscreteInfoEstimator{I}
    measure::I
end
JackknifeEstimator() = JackknifeEstimator(Shannon())

function information(hest::JackknifeEstimator{<:Shannon}, pest::ProbabilitiesEstimator, x)
    h_naive = information(PlugIn(hest.measure), pest, x)
    N = length(x)
    h_jackknifed = zeros(N)
    for i in eachindex(x)
        idxs = setdiff(1:N, i)
        xᵢ = @views x[idxs]
        h_jackknifed[i] = information(PlugIn(hest.measure), pest, xᵢ)
    end
    return N * h_naive - (N - 1)/N * sum(h_jackknifed)
end
