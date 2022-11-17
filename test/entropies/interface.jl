x = ones(3)
@test_throws MethodError entropy(x, 0.1)
