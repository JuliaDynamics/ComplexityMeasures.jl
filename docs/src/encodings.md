# Encodings

## Encoding API

Some probability estimators first "encode" input data into an intermediate representation indexed by the positive integers. This intermediate representation is called an "encoding" and its API is defined by the following:

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

