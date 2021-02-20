using DelayEmbeddings, SparseArrays

export 
    TransferOperator, # the probabilities estimator
    InvariantMeasure,
    transfermatrix,
    invariantmeasure

"""
    TransferOperator(ϵ::RectangularBinning) <: BinningProbabilitiesEstimator

A probability estimator based on binning data into rectangular boxes dictated by 
the binning scheme `ϵ`, then approxmating the transfer (Perron-Frobenius) operator 
over the bins, the taking the invariant measure associated with that transfer operator 
as the bin probabilities. Assumes that the input data are sequential.

This implementation follows the grid estimator approach in Diego et al. (2019)[^Diego2019].

## Description

The transfer operator ``P^{N}``is computed as an `N`-by-`N` matrix of transition 
probabilities between the states defined by the partition elements, where `N` is the 
number of boxes in the partition that is visited by the orbit/points. 

If  ``\\{x_t^{(D)} \\}_{n=1}^L`` are the ``L`` different ``D``-dimensional points over 
which the transfer operator is approximated, ``\\{ C_{k=1}^N \\}`` are the ``N`` different 
partition elements (as dictated by `ϵ`) that gets visited by the points, and
 ``\\phi(x_t) = x_{t+1}``, then

```math
P_{ij} = \\dfrac
{\\#\\{ x_n | \\phi(x_n) \\in C_j \\cap x_n \\in C_i \\}}
{\\#\\{ x_m | x_m \\in C_i \\}},
```

where ``\\#`` denotes the cardinal. The element ``P_{ij}`` thus indicates how many points 
that are initially in box ``C_i`` end up in box ``C_j`` when the points in ``C_i`` are 
projected one step forward in time. Thus, the row ``P_{ik}^N`` where 
``k \\in \\{1, 2, \\ldots, N \\}`` gives the probability 
of jumping from the state defined by box ``C_i`` to any of the other ``N`` states. It 
follows that ``\\sum_{k=1}^{N} P_{ik} = 1`` for all ``i``. Thus, ``P^N`` is a row/right 
stochastic matrix.

### Invariant measure estimation from transfer operator

The left invariant distribution ``\\mathbf{\\rho}^N`` is a row vector, where 
``\\mathbf{\\rho}^N P^{N} = \\mathbf{\\rho}^N``. Hence, ``\\mathbf{\\rho}^N`` is a row 
eigenvector of the transfer matrix ``P^{N}`` associated with eigenvalue 1. The distribution 
``\\mathbf{\\rho}^N`` approximates the invariant density of the system subject to the 
partition `ϵ`, and can be taken as a probability distribution over the partition elements.

In practice, the invariant measure ``\\mathbf{\\rho}^N`` is computed using 
[`invariantmeasure`](@ref), which also approximates the transfer matrix. The invariant distribution
is initialized as a length-`N` random distribution which is then applied to ``P^{N}``. 
The resulting length-`N` distribution is then applied to ``P^{N}`` again. This process 
repeats until the difference between the distributions over consecutive iterations is 
below some threshold. 

## Probability and entropy estimation

- `probabilities(x::AbstractDataset, est::TransferOperator{RectangularBinning})` estimates 
    probabilities for the bins defined by the provided binning (`est.ϵ`)
- `genentropy(x::AbstractDataset, est::TransferOperator{RectangularBinning})` does the same, 
    but computes generalized entropy using the probabilities.


See also: [`RectangularBinning`](@ref), [`invariantmeasure`](@ref).

[^Diego2019]: Diego, D., Haaga, K. A., & Hannisdal, B. (2019). Transfer entropy computation using the Perron-Frobenius operator. Physical Review E, 99(4), 042212.
"""
struct TransferOperator{R} <: BinningProbabilitiesEstimator
    ϵ::R
    
    function TransferOperator(ϵ::R) where R #<: RectangularBinning
        new{R}(ϵ)
    end
end
struct TransferOperatorGenerator{E <: TransferOperator, X, A}
    method::E # estimator with its input parameters
    pts::X    # the phase space / reconstruted state space points
    init::A   # pre-initialized things that speed up estimation process
end


"""
    transopergenerator(pts, method::TransferOperator) → to::TransferOperatorGenerator

Initialize a generator that creates transfer operators on demand, based on the given `method`.
This is efficient, because some things can be initialized and reused.

To approximate a transfer operator, call `to` as a function with the relevant arguments.

```julia
to = transopergenerator(x, TransferOperator(RectangularBinning(5)))
for i in 1:1000
    s = to()
    # do stuff with s and or x
    result[i] = stuff
end
```
"""
function transopergenerator end

function transferoperator end

function invariantmeasure end 

function transfermatrix end

function transferoperator(pts, method::TransferOperator)
    to = transopergenerator(pts, method)
    to()    
end

""" 
    InvariantMeasure(to, ρ)

Minimal return struct for [`invariantmeasure`](@ref) that contains the estimated invariant 
measure `ρ`, as well as the transfer operator `to` from which it is computed (including 
bin information).

See also: [`invariantmeasure`](@ref).
""" 
struct InvariantMeasure{T}
    to::T
    ρ::Probabilities

    function InvariantMeasure(to::T, ρ) where T
        new{T}(to, ρ)
    end
end
