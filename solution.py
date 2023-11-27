
def sum_function(a, b):
    if isinstance(a, list) or isinstance(b, list):
        raise TypeError("Both inputs must be of the same type.")
    return a + b

result = sum_function("a", ["b", "b", "b"])
print(result)
