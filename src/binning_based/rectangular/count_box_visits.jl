
export joint_visits, marginal_visits

# Count bin visits by counting equal encoded bins. Either for all dimensions at a 
# time (`joint_visits`), or for a subset of dimensions (`marginal_visits`).

"""
    joint_visits(points, binning_scheme::RectangularBinning) → Vector{Vector{Int}}

Determine which bins are visited by `points` given the rectangular binning
scheme `ϵ`. Bins are referenced relative to the axis minima, and are 
encoded as integers, such that each box in the binning is assigned a
unique integer array (one element for each dimension). 

For example, if a bin is visited three times, then the corresponding 
integer array will appear three times in the array returned.

See also: [`marginal_visits`](@ref), [`encode`](@ref).

# Example 

```julia
using DelayEmbeddings, Entropies

pts = Dataset([rand(5) for i = 1:100]);
joint_visits(pts, RectangularBinning(0.2))
```
"""
function joint_visits(points, binning_scheme::RectangularBinning)
    axis_minima, box_edge_lengths = get_minima_and_edgelengths(points, binning_scheme)
    encode(points, axis_minima, box_edge_lengths)
end

"""
    marginal_visits(points, binning_scheme::RectangularBinning, dims) → Vector{Vector{Int}}

Determine which bins are visited by `points` given the rectangular binning
scheme `ϵ`, but only along the desired dimensions `dims`. Bins are referenced 
relative to the axis minima, and are encoded as integers, such that each box 
in the binning is assigned a unique integer array (one element for each 
dimension in `dims`). 

For example, if a bin is visited three times, then the corresponding 
integer array will appear three times in the array returned.

See also: [`joint_visits`](@ref), [`encode`](@ref).

# Example

```julia
using DelayEmbeddings, Entropies
pts = Dataset([rand(5) for i = 1:100]);

# Marginal visits along dimension 3 and 5
marginal_visits(pts, RectangularBinning(0.3), [3, 5])

# Marginal visits along dimension 2 through 5
marginal_visits(pts, RectangularBinning(0.3), 2:5)
```
"""
function marginal_visits(points, binning_scheme::RectangularBinning, dims)
    axis_minima, box_edge_lengths = get_minima_and_edgelengths(points, binning_scheme)
    dim = length(axis_minima)
    if length(dims) == 1
        dims = [dim]
    end
    if sort(collect(dims)) == sort(collect(1:dim))
        joint_visits(points, binning_scheme)
    else
        [encode(pt, axis_minima, box_edge_lengths)[dims] for pt in points]
    end
end

"""
    marginal_visits(joint_visits, dims) → Vector{Vector{Int}}

If joint visits have been precomputed using [`joint_visits`](@ref), marginal 
visits can be returned directly without providing the binning again 
using the `marginal_visits(joint_visits, dims)` signature.

See also: [`joint_visits`](@ref), [`encode`](@ref).

# Example 

```
using DelayEmbeddings, Entropies
pts = Dataset([rand(5) for i = 1:100]);

# First compute joint visits, then marginal visits along dimensions 1 and 4
jv = joint_visits(pts, RectangularBinning(0.2))
marginal_visits(jv, [1, 4])

# Marginals along dimension 2
marginal_visits(jv, 2)
```
"""
function marginal_visits(joint_visits, dims)
    [bin[dims] for bin in joint_visits]
end