using StaticArrays

function update_x̂!(x̂, X, ntot, i, j, d)
    Q = view(X, j:j+d-1, i:i+d-1)

    for i in 1:ntot
        x̂[i] = Q[i]
    end
end

function update_motif!(motifs, k, motif)
    motifs[k] = motif
end

function update_motifs!(motifs, x̂, x̂_sorted, X, d)
    ny, nx = size(X)
    ntot = d * d

    rx = 1:nx-d+1 # columns
    ry = 1:ny-d+1 # rows
    k = 1
    for i in rx
        for j in ry
            update_x̂!(x̂, X, ntot, i, j, d)
            sortperm!(x̂_sorted, x̂)
            update_motif!(motifs, k, Entropies.encode_motif(x̂_sorted))
            k += 1
        end
    end
    return motifs
end

function probabilitygenerator(x::AbstractArray{T, 2}, method::SymbolicPermutation2D,
    rng = Random.default_rng()) where T
    d = method.m
    ntot = d * d
    # Picking the first `ntot` elements of x to infer type from `x`, so that we can
    # use non-numeric types as input too.
    x̂ = x[1:ntot]
    x̂_sorted = [i for i = 1:ntot]

    # Pre-allocate motif vector.
    ny, nx = size(x)

    # Picking sub-matrices means there are fewer than `ny`/`nx` rows/columns
    # to pick from.
    nr = length(1:nx-d+1) # rows
    nc = length(1:ny-d+1) # columns
    motifs = zeros(Int, nr * nc)

    init = (
        x̂ = x̂,
        x̂_sorted = x̂_sorted,
        motifs = motifs,
    )
    return ProbabilityGenerator(method, x, init, rng)
end

function entropygenerator(x::AbstractArray{T, 2}, method::SymbolicPermutation2D,
        rng = Random.default_rng()) where T
    pg = probabilitygenerator(x, method, rng)
    init = ()
    return EntropyGenerator(method, pg, x, init, rng)
end

function (eg::ProbabilityGenerator{<:SymbolicPermutation2D})(; kwargs...)
    throw(ArgumentError("Missing 2D array input to probability generator"))
end

function (eg::ProbabilityGenerator{<:SymbolicPermutation2D})(X::AbstractArray{T, 2}) where T
    d = eg.method.m
    x̂, x̂_sorted, motifs = getfield.(
        Ref(eg.init),
        (:x̂, :x̂_sorted, :motifs)
    )
    # Loop through the d*d-sized submatrices and symbolize them, one by one, into
    # the pre-allocated `motifs` vector.
    update_motifs!(motifs, x̂, x̂_sorted, X, d)

    return Probabilities(_non0hist(motifs) ./ length(motifs))
end


function (eg::EntropyGenerator{<:SymbolicPermutation2D, PROBGEN})(X::AbstractArray{T, 2};
        base = MathConstants.e, normalize = true, q = 1) where {PROBGEN, T}

    ps = eg.probability_generator(X)
    d = eg.probability_generator.method.m
    if normalize
        # For `d > 4`, we need BigInt to compute the factorial of the product.
        nc = convert(Float64, log(base, factorial(BigInt(d) * BigInt(d))))
        return float(genentropy(ps, base = base, q = q) / nc)
    else
        return genentropy(ps, base = base, q = q)
    end
end

function probabilities(x::AbstractArray{T, 2}, est::SymbolicPermutation2D) where T
    pg = probabilitygenerator(x, est)
    return pg(x)
end

function genentropy(x::AbstractArray{T, 2}, est::SymbolicPermutation2D;
        q = 1, base = MathConstants.e, normalize = true) where T
    eg = entropygenerator(x, est)
    eg(x, base = base, q = q, normalize = normalize)
end
