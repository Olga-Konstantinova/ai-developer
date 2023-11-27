# This is a Python code snippet with an intentional error

def calculate_average(numbers):
    if not numbers:
        raise ValueError("Input list is empty")  # Error: Division by zero not handled

    total_sum = sum(numbers)
    average = total_sum / len(numbers)  # Error: Division by zero if 'numbers' is an empty list
    return average

# Function to square each element in a list
def square_elements(lst):
    return [element ** 2 for element in lst]

# Function to calculate the sum of even numbers in a list
def sum_even_numbers(lst):
    return sum([num for num in lst if num % 2 == 0])

# Function to calculate the product of elements in a list
def calculate_product(lst):
    product = 1
    for num in lst:
        product *= num
    return product

# Function to generate a Fibonacci sequence up to a specified number of terms
def fibonacci_sequence(n):
    fib_sequence = [0, 1]
    while len(fib_sequence) < n:
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence

# Function to perform matrix multiplication
def matrix_multiply(matrix1, matrix2):
    result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]

    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result


# numbers_list = []
# average_result = calculate_average(numbers_list)
# print("Average:", average_result)


def sum_function(a, b):
    return a + b

sum_function("a", ["b", "b", "b"])
