# Encodings

## Encodings API

Some probability estimators first "encode" input data into an intermediate representation indexed by the positive integers. This intermediate representation is called an "encoding".

The encodings API is defined by:

- [`Encoding`](@ref)
- [`encode`](@ref)
- [`decode`](@ref)

```@docs
Encoding
encode
decode
```

## Available encodings

```@docs
OrdinalPatternEncoding
GaussianCDFEncoding
RectangularBinEncoding
```

