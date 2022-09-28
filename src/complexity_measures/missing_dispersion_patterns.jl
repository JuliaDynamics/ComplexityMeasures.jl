export missing_dispersion

"""
    missing_dispersion(x::AbstractVector{T}; est::Dispersion = Dispersion()) → N::Real

Calculate the number of missing dispersion patterns (NMDP; Zhou et al., 2022) resulting
from symbolising `x` and embedding the resulting symbol sequence (using the parameters of
`est`) into "dispersion patterns" (each embedding vector is a dispersion pattern).

The state vectors of the resulting embedding are called "dispersion patterns", and there
are `L = alphabet_length(est)` possible patterns. The number of missing dispersion patterns
is simply `L - N`, where `N` is the number of patterns actually occurring.

Used to detect nonlinearity in a time series by comparing the NMDP `x` to NMDP values for
an ensemble of IAAFT surrogates of `x`.

See also: [`Dispersion`](@ref), [`alphabet_length`](@ref), [`reverse_dispersion`](@ref).

[Zhou2022]: Zhou, Q., Shang, P., & Zhang, B. (2022). Using missing dispersion patterns
    to detect determinism and nonlinearity in time series data. Nonlinear Dynamics, 1-20.
"""
function missing_dispersion(x::AbstractVector{T}; est::Dispersion = Dispersion();
        normalize::Bool = false) where T

    symbols = _symbolize_for_dispersion(x, est)

    n_occuring = length(Set(symbols))
    L = alphabet_length(est)
    n_not_occurring = L - n_occuring

    # Converting to float ensures return type stability.
    if (normalize)
        return float(n_not_occurring / L)
    else
        return float(n_not_occurring)
    end
end
