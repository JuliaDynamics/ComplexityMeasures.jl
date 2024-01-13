"""
    relevant_fieldnames(x::T) → names::Vector{Symbol}

Internal method that returns the relevant field names to be printed for type `T`.
For example, for `Encodings`, the implementation is simply

```julia
relevant_fieldnames(e::Encoding) = fieldnames(typeof(e))
```

Individual types can override this method if special printing is desired.
"""
relevant_fieldnames(x) = fieldnames(typeof(x))

"""
    type_printcolor(x) → color::Symbol

The color in which to print the type name for type `typeof(x)`.

Can be used to distinguish different types, so that nested printing looks better (e.g.
`Encoding`s inside `OutcomeSpace`s).
"""
type_printcolor(x) = :normal

"""
    type_field_printcolor(x) → color::Symbol

The color in which to print the field names for a type of type `typeof(x)`.

Used in combination with [`type_printcolor`](@ref) to make nested printing look better.
"""
type_field_printcolor(x) = :grey # fallback

"""
    hidefields(::Type{T})

Returns an iterable of symbols incidating fields to hide for instances of type `T`.
"""
hidefields(x) = Symbol[]

"""
    PrintComponent
    PrintComponent(s; color::Union{Symbol,Int} = :normal,
        bold::Bool = false, underline::Bool = false, blink::Bool = false,
        hidden::Bool = false, reverse::Bool = false)

Stores a string `s` and instructions for how it shall be printed. 
    
`PrintComponent`s are intended for use with `printstyled`.
"""
struct PrintComponent{S<:AbstractString}
    s::S
    color::Union{Symbol,Int}
    bold::Bool
    underline::Bool
    blink::Bool
    hidden::Bool
    reverse::Bool

    function PrintComponent(s::S; color::Union{Symbol,Int} = :normal,
        bold::Bool = false, underline::Bool = false, blink::Bool = false,
        hidden::Bool = false, reverse::Bool = false) where S
        new{S}(s, color, bold, underline, blink, hidden, reverse) 
    end
end
Base.show(io::IO, x::PrintComponent) = printstyled(io, x.s; 
    color = x.color, bold = x.bold, underline = x.underline, blink = x.blink,
    hidden = x.hidden, reverse = x.reverse)

function Base.show(io::IO, x::Vector{<:PrintComponent}) 
    for component in x
        show(io, component)
    end
end

# The entire purpose for this type is so that we can print nested formatted types.
struct EntireComponent{T}
    s::Vector{T}
end

function Base.show(io::IO, x::EntireComponent)
    for component in x.s
        show(io, component)
    end
end


struct PrintComponents{T}
    x::Vector{T}
end
function Base.show(io::IO, x::PrintComponents) 
    for component in x.x
        show(io, component)
    end
end

# Modify here if more of our abstract types should be considered.
our_abstract_types = [Encoding, 
    OutcomeSpace, 
    ProbabilitiesEstimator, 
    InformationMeasure,
    DiscreteInfoEstimator,
    DifferentialInfoEstimator,
    ComplexityEstimator
]

type_printcolor(x::Type{<:Encoding}) = :black
type_printcolor(x::Type{<:OutcomeSpace}) = :blue
type_printcolor(x::Type{<:ProbabilitiesEstimator}) = :light_green
type_printcolor(x::Type{<:InformationMeasure}) = :yellow
type_printcolor(x::Type{<:InformationMeasureEstimator}) = :light_magenta
type_printcolor(x::Type{<:ComplexityEstimator}) = :red

type_field_printcolor(x::Type{<:Encoding}) = :light_black
type_field_printcolor(x::Type{<:OutcomeSpace}) = :light_blue
type_field_printcolor(x::Type{<:ProbabilitiesEstimator}) = :light_green
type_field_printcolor(x::Type{<:InformationMeasure}) = :light_yellow
type_field_printcolor(x::Type{<:InformationMeasureEstimator}) = :light_magenta
type_field_printcolor(x::Type{<:ComplexityEstimator}) = :black

function single_print_component!(v, name, fieldval, x; fieldcol = :grey)
    # Field names are colored as a weaker variant of the parent type color.
    push!(v, PrintComponent("$name"; bold=false, color=fieldcol))

    # Use standard formatting for the rest.
    push!(v, PrintComponent(" = "; bold=false, color = :default))
    if any(typeof(fieldval) <: T for T in our_abstract_types)
        # If we want more aggressive coloring, switch to this:
        #custom_fieldcolor = type_field_printcolor(typeof(fieldval))
        #comps = printcomponents(fieldval; custom_fieldcolor)
        comps = printcomponents(fieldval)
        push!(v, EntireComponent(comps))
    else
        if any(typeof(x) <: T for T in our_abstract_types)
            custom_fieldcolor = type_field_printcolor(typeof(fieldval))
        else
            custom_fieldcolor = :grey
        end
        push!(v, PrintComponent("$fieldval"; bold=false, color=fieldcol))
    end
end

# TODO: extra explicity type paremters for certain types, e.g. {m} for OrdinalPatterns.

const tabSpace = " "

const FANCY_PRINTABLE = Union{PrintComponent, EntireComponent}
function printcomponents(x; custom_fieldcolor = :default, compact = true)
    v = Vector{FANCY_PRINTABLE}(undef, 0)
    T = typeof(x)
    N = T.name.name
    push!(v, PrintComponent("$N("; color = type_printcolor(T), bold = true))
    if !compact
        push!(v, PrintComponent("\n$tabSpace"; hidden=false))
    end
    shownames = [name for name in relevant_fieldnames(x) if !(name in hidefields(T))]
    
    for (i, name) in enumerate(shownames)
        single_print_component!(v, name, getfield(x, name), x; 
            fieldcol = custom_fieldcolor)
        if i < length(shownames)
            if compact
                push!(v, PrintComponent(", "; color=:normal))
            else
                push!(v, PrintComponent("\n$tabSpace"; color=:normal))
            end
        end
    end
    if !compact
        push!(v, PrintComponent("\n"; hidden=false))
    end
    push!(v, PrintComponent(")"; color = type_printcolor(T), bold = true))
    

    # Don't convert to PrintComponents yet, because that will lead to a nested mess.
    # We convert inside the extension to `Base.show` instead (see below).
    return v
end

for S in our_abstract_types
    # For standalone printing of the type.
    @eval function Base.show(io::IO, ::MIME"text/plain", x::$S)
        T = typeof(x)
        shownames = [name for name in relevant_fieldnames(x) if !(name in hidefields(T))]
        if length(shownames) <= 1
            compact = true
        else
            compact = false
        end
        p = PrintComponents(printcomponents(x, compact = compact))
        show(io, p)
    end
    # For compact printing (inside vectors, tuples etc)
    @eval function Base.show(io::IO, x::$S)
        p = PrintComponents(printcomponents(x, compact = true))
        show(io, p)
    end
end

