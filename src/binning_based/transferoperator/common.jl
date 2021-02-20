using DelayEmbeddings 
using Entropies 
import LinearAlgebra: norm

export 
    transferoperator, 
    TransferOperator,
    TransferOperatorApproximation,
    TransferMatrix,
    invariantmeasure, 
    InvariantMeasure,
    transitioninfo,
    TransitionInfo

"""
    TransferOperator(ϵ::Union{RectangularBinning, SimplexPoint, SimplexExact}) <: BinningProbabilitiesEstimator

A probability estimator based on binning data into rectangular boxes 
(when `ϵ` is a [`RectangularBinning`](@ref)) or simplices (when `ϵ` is a 
[`SimplexExact`](@ref) or `ϵ` is a [`SimplexPoint`](@ref)). 

To use triangulation-based estimators, run `using Simplices` after `using Entropies`.

The transfer (Perron-Frobenius) operator is approximated over the bins, using different 
methods depending on the type of partition chosen. The invariant measure associated with 
the approximated transfer operator is taken the bin probabilities. 

# Description (rectangular binning)

This implementation follows the grid estimator approach in Diego et al. (2019)[^Diego2019].

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

For a detailed description of the triangulation estimators, see 
Diego et al. (2019)[^Diego2019].

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

- `probabilities(x::AbstractDataset, est::TransferOperator)` estimates 
    probabilities for the bins defined by the provided binning (`est.ϵ`)
- `genentropy(x::AbstractDataset, est::TransferOperator)` does the same, 
    but computes generalized entropy using the probabilities.


See also: [`RectangularBinning`](@ref), [`SimplexPoint`](@ref), [`SimplexExact`](@ref), 
[`invariantmeasure`](@ref)

## Examples

Here, we create three different transfer operator-based estimators.

```@example
using Entropies
# A rectangular binning is suited for datasets with a large number of points
est_rect = TransferOperator(RectangularBinning(5))

# A triangulated binning, using approximate simplex intersections, is also possible for 
# datasets with not too many points (say, <1000 points). If so, we must first import 
# the Simplices.jl package.
using Simplices
est_point = TransferOperator(SimplexPoint())

# For datasets with few points, say <100 points, exact simplex intersections may also 
# be computationally feasible.
est_exact = TransferOperator(SimplexExact())
```

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


function Base.show(io::IO, DT::TransferOperatorGenerator{E, X, A}) where {E, X, A}
    summary = "TransferOperatorGenerator{method: $E, pts: $X, init: $A}"
    println(io, summary)
end


function transferoperator end

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

function transferoperator(pts, method::TransferOperator)
    to = transopergenerator(pts, method)
    to()    
end

"""
    TransferOperatorApproximation(generator, transfermatrix, params)

A return struct containing the `generator` of the transfer operator approximation (an 
instance of `TransferOperator{R} where R`; if `R` is a `RectangularBinning`, then a 
rectangular binning was used, whereas if `R` is `SimplexPoint` or `SimplexExact`, then 
a triangulated binning as used). 

The `transfermatrix` is an `N`-by-`N` matrix approximation to the transfer operator, 
subject to the partition given by `generator` (details about the partition are available
in the `generator.init` field), computed over some set of sequentially ordered points. 

# Parameters

The parameters of the approximation is given to `params`. 

When `R <: RectangularBinning`, parameters stored are as follows.

- The `mini` and `edgelengths` parameters provide the minima and box edge lengths along each 
    coordinate axis, as determined by applying `ϵ` to the points. 
- The coordinates of the (leftmost, if axis is ordered low-high) box corners are given in 
    `bins`. Only bins actually visited by the points are considered, and `bins` give the 
    coordinates of these bins. The element `bins[i]` correspond to the `i`-th state of the 
    system, which corresponds to the `i`-th column/row of `transfermatrix`.
- `sort_idxs` contains the indices that would sort the input points. 
- `visitors` is a vector of vectors, where `visitors[i]` contains the indices of the 
    (sorted) points that visits `bins[i]`.

When `R <: SimplexPoint`, parameters stored are the simplex subsampling parameters.
When `R <: SimplexExact`, parameters stored are the simplex intersection tolerances. 

See also: [`RectangularBinning`](@ref), [`SimplexPoint`](@ref), [`SimplexExact`](@ref).
"""
struct TransferOperatorApproximation{G<:TransferOperator, T}
    generator::TransferOperatorGenerator{G}
    transfermatrix::T
    params
end

function Base.show(io::IO, DT::TransferOperatorApproximation{G, T}) where {G, T}
    summary = "TransferOperatorApproximation{generator: $G, transfer matrix: $T}"
    println(io, summary)
end


"""
    TransitionInfo(transfermatrix, bins)

Structure that holds the `transfermatrix` of transition probabilities obtained from a 
transfer operator approximation, as well as the `bins` of the partition.

- If `transfermatrix` was obtained using a rectangular binning, then `bins` is a 
    two-element named tuple with fields `bins` (the left-most coordinates of the bins) and 
    `edgelengths` (the edge lengths along each coordinate axis).
- If `transfermatrix` was obtained using a triangulated binning, then `bins` is a 
    two-element named tuple with fields `pts` (the points of the triangulation), and 
    `triang`, the indices of the vertices of all simplices. Here, `pts[triang[i]]` is a 
    vector of vertices for the `i`-th simplex of the triangulation.

`bins[i]` corresponds to the i-th row/column of `transfermatrix.` Thus, the entry 
`transfermatrix[i, j]` is the probability of jumping from the state defined by `bins[i]` 
to the state defined by `bins[j]`.

See also [`transitioninfo`](@ref).
"""
struct TransitionInfo{T, B}
    transfermatrix::T
    bins::B
end


function Base.show(io::IO, info::TransitionInfo{T, B}) where {T, B}
    summary = "TransitionInfo{transfermatrix: $T, bins: $B}"
    println(io, summary)
end

"""
    transitioninfo(to::TransferOperatorApproximation) → TransitionInfo
    transitioninfo(to::InvariantMeasure) → TransitionInfo

Convenience method to get the transfer matrix/operator and the corresponding bins from 
pre-computed quantities. 

See also: [`TransitionInfo`](@ref), [`TransferOperatorApproximation`](@ref).
"""
function transitioninfo end

""" 
    InvariantMeasure(to::TransferOperatorApproximation, ρ::Probabilities)

Minimal return struct for [`invariantmeasure`](@ref) that contains the estimated invariant 
measure `ρ`, as well as the transfer operator `to` from which it is computed (including 
bin information).

See also: [`invariantmeasure`](@ref).
""" 
struct InvariantMeasure{T<:TransferOperatorApproximation, P<:Probabilities}
    to::T
    ρ::P

    function InvariantMeasure(to::T, ρ::P) where {T, P}
        new{T, P}(to, ρ)
    end
end

function Base.show(io::IO, DT::InvariantMeasure{T, P}) where {T, P}
    summary = "InvariantMeasure{transfer operator approximation: $T, probabilities: $P}"
    println(io, summary)
end

import LinearAlgebra: norm
"""
    invariantmeasure(x::AbstractDataset, to::TransferOperator{R})
        where {R <: Union{RectangularBinning, SimplexExact, SimplexPoint}} 
        → iv::InvariantMeasure

Estimate an invariant measure over the points in `x` based on binning the data into 
rectangular boxes dictated by the binning scheme `ϵ`, then approximate the transfer 
(Perron-Frobenius) operator over the bins. From the approximation to the transfer operator, 
compute an invariant distribution over the bins. Assumes that the input data are sequential.

Details on the estimation procedure is found the [`TransferOperator`](@ref) docstring.

## Example 

```julia
using DynamicalSystems, Plots, Entropies
D = 4
ds = Systems.lorenz96(D; F = 32.0)
N, dt = 20000, 0.1
orbit = trajectory(ds, N*dt; dt = dt, Ttr = 10.0)

# Estimate the invariant measure over some rectangular coarse graining of the orbit.
iv = invariantmeasure(orbit, RectangularBinning(15))

# Get the probabilities and bins 
invariantmeasure(iv)
```

## Probabilities and bin information

    invariantmeasure(iv::InvariantMeasure) → (ρ::Probabilities, bins::Vector{<:SVector})

From a pre-computed invariant measure, return the probabilities and associated bins. 
The element `ρ[i]` is the probability of visitation to the box `bins[i]`. Analogous to 
[`binhist`](@ref). 


!!! hint "Transfer operator approach vs. naive histogram approach"

    Why bother with the transfer operator instead of using regular histograms to obtain 
    probabilities? 
    
    In fact, the naive histogram approach and the 
    transfer operator approach are equivalent in the limit of long enough time series 
    (as ``n \\to \\intfy``), which is guaranteed by the ergodic theorem. There is a crucial
    difference, however:
    
    The naive histogram approach only gives the long-term probabilities that 
    orbits visit a certain region of the state space. The transfer operator encodes that 
    information too, but comes with the added benefit of knowing the *transition 
    probabilities* between states (see [`transfermatrix`](@ref)). 

See also: [`InvariantMeasure`](@ref).
"""
function invariantmeasure(to::TransferOperatorApproximation; 
        N::Int = 200, tolerance::Float64 = 1e-8, delta::Float64 = 1e-8)
    
    TO = to.transfermatrix
    #=
    # Start with a random distribution `Ρ` (big rho). Normalise it so that it
    # sums to 1 and forms a true probability distribution over the partition elements.
    =#
    Ρ = rand(Float64, 1, size(to.transfermatrix, 1))
    Ρ = Ρ ./ sum(Ρ, dims = 2)

    #=
    # Start estimating the invariant distribution. We could either do this by
    # finding the left-eigenvector of M, or by repeated application of M on Ρ
    # until the distribution converges. Here, we use the latter approach,
    # meaning that we iterate until Ρ doesn't change substantially between
    # iterations.
    =#
    distribution = Ρ * to.transfermatrix

    distance = norm(distribution - Ρ) / norm(Ρ)

    check = floor(Int, 1 / delta)
    check_pts = floor.(Int, transpose(collect(1:N)) ./ check) .* transpose(collect(1:N))
    check_pts = check_pts[check_pts .> 0]
    num_checkpts = size(check_pts, 1)
    check_pts_counter = 1

    counter = 1
    while counter <= N && distance >= tolerance
        counter += 1
        Ρ = distribution

        # Apply the Markov matrix to the current state of the distribution
        distribution = Ρ * to.transfermatrix

        if (check_pts_counter <= num_checkpts &&
           counter == check_pts[check_pts_counter])

            check_pts_counter += 1
            colsum_distribution = sum(distribution, dims = 2)[1]
            if abs(colsum_distribution - 1) > delta
                distribution = distribution ./ colsum_distribution
            end
        end

        distance = norm(distribution - Ρ) / norm(Ρ)
    end
    distribution = dropdims(distribution, dims = 1)

    # Do the last normalisation and check
    colsum_distribution = sum(distribution)

    if abs(colsum_distribution - 1) > delta
        distribution = distribution ./ colsum_distribution
    end

    # Find partition elements with strictly positive measure.
    δ = tolerance/size(to.transfermatrix, 1)
    inds_nonzero = findall(distribution .> δ)

    # Extract the elements of the invariant measure corresponding to these indices
    return InvariantMeasure(to, Probabilities(distribution))
end


function invariantmeasure(pts, method::TransferOperator{<:R}; kwargs...) where R
    to_approximation = transferoperator(pts, method; kwargs...)
    invariantmeasure(to_approximation)
end

probabilities(iv::InvariantMeasure) = iv.ρ
probabilities(iv::TransferOperatorApproximation) = probabilities(invariantmeasure(iv))

function probabilities(pts, method::TransferOperator{<:R}; kwargs...) where R
    to_approximation = transferoperator(pts, method; kwargs...)
    invariantmeasure(to_approximation).ρ
end