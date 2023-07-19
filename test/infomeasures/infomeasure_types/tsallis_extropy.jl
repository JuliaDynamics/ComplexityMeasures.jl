# Constructors
m1 = TsallisExtropy(q = 2)
m2 = TsallisExtropy(2)
@test m1.q == 2
@test m2.q == 2

# Minimized for one-element distributions, where there is total order.
x = [0.1, 0.1, 0.1]
@test information(TsallisExtropy(q = 2), CountOccurrences(), x) == 0.0
@test information(TsallisExtropy(2), CountOccurrences(), x) == 0.0

@test information_normalized(TsallisExtropy(q = 2), CountOccurrences(), x) == 0.0


# Example 3.5 from Xue & Deng (2023) (equivalence of Tsallis entropy/information for
# two-element distributions)
x = [0.2, 0.4, 0.4]
h = information(Tsallis(; q = 2), CountOccurrences(), x)
j = information(TsallisExtropy(; q = 2), CountOccurrences(), x)
@test 2h ≈ 2j ≈ 8/9

# Example 3.4 from Xue & Deng (2023) (equivalence of Tsallis entropy/information for
# two-element distributions)
x = [0.2, 0.4]
h = information(Tsallis(; q = 2), CountOccurrences(), x)
j = information(TsallisExtropy(; q = 2), CountOccurrences(), x)
@test 2h ≈ 2j ≈ 1.0

# Example 3.9 from Xue & Deng (2023)
x = [0.2, 0.2, 0.4, 0.4, 0.5, 0.5]
j = information(TsallisExtropy(; q = 2), CountOccurrences(), x)
@test j ≈ 2/3

# In general, normalized Tsallis information should be maximized (i.e. be equal to 1) for a
# uniform distribution.
x = [0.2, 0.2, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6]
jn = information_normalized(TsallisExtropy(; q = 2), CountOccurrences(), x)
@test jn == 1
