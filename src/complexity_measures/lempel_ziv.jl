export LempelZiv76

"""
    LempelZiv76 <: ComplexityEstimator
    LempelZiv76()

The Lempel-Ziv, or `LempelZiv76`, complexity measure (Lempel & Ziv, 1976) [^LempelZiv1976],
which is used with [`complexity`](@ref) and [`complexity_normalized`](@ref).

For results to be comparable across sequences with different length, use the normalized
version. Normalized LempelZiv76-complexity is implemented as given in Amigó et al.
(2004)[^Amigó2004]. The normalized measure is close to zero for very regular signals, while
for random sequences, it is close to 1 with high probability[^Amigó2004]. Note: the
normalized LempelZiv76 complexity can be higher than 1[^Amigó2004].

The `LempelZiv76` measure applies only to binary sequences, i.e. sequences with a
two-element alphabet (precisely two distinct outcomes). For performance optimization,
we do not check the number of unique elements in the input. If your input sequence is not
binary, you must [`encode`](@ref) it first using one of the implemented [`Encoding`](@ref)
schemes (or encode your data manually).

[^LempelZiv1976]:
    Lempel, A., & Ziv, J. (1976). On the complexity of finite sequences. IEEE Transactions
    on information theory, 22(1), 75-81.
[^Amigó2004]:
    Amigó, J. M., Szczepański, J., Wajnryb, E., & Sanchez-Vives, M. V. (2004). Estimating
    the entropy rate of spike trains via Lempel-Ziv complexity. Neural Computation, 16(4),
    717-736.
"""
struct LempelZiv76 <: ComplexityEstimator end

function complexity(::LempelZiv76, x::AbstractArray{T, N}) where {T, N}
    # The implementation here is taken directly from
    # https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv_complexity
    i = 0 # `i = p − 1`, `p is the pointer
    c = 1 # The complexity counter
    u = 1 # Length of the current prefix
    v = 1 # Length of the current component for the current pointer `p``
    vmax = v
    n = length(x)
    @inbounds while u + v <= n
        if x[i + v] == x[u + v]
            v += 1
        else
            vmax = max(v, vmax)
            i += 1
            if i == u  # all pointers have been treated
                c += 1
                u += vmax
                v = 1
                i = 0
                vmax = v
            else
                v = 1
            end
        end
    end
    if v != 1
        c += 1
    end
    return c
end

# Formula from Amigó et al. (2004)
function complexity_normalized(est::LempelZiv76, x)
    n = length(x)
    c = complexity(est, x)
    return c / (n / log2(n))
end
