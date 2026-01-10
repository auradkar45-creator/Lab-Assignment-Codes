#Set-A, BL.SC.U4AIE24005

def count_pairs_with_sum_10(numbers):
    count = 0
    length = len(numbers)
    for i in range(length):
        for j in range(i + 1, length):
            if numbers[i] + numbers[j] == 10:
                count += 1
    return count


def find_range_of_list(numbers):
    if len(numbers) < 3:
        return "Range determination not possible"
    
    minimum = numbers[0]
    maximum = numbers[0]

    for num in numbers:
        if num < minimum:
            minimum = num
        if num > maximum:
            maximum = num

    return maximum - minimum


def matrix_power(matrix, power):
    size = len(matrix)

    #Identity matrix
    result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]

    def multiply(a, b):
        product = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    product[i][j] += a[i][k] * b[k][j]
        return product

    for _ in range(power):
        result = multiply(result, matrix)

    return result


def highest_occurring_character(text):
    frequency = {}
    for char in text.lower():
        if char.isalpha():
            frequency[char] = frequency.get(char, 0) + 1

    max_char = None
    max_count = 0

    for char in frequency:
        if frequency[char] > max_count:
            max_count = frequency[char]
            max_char = char

    return max_char, max_count


def mean_median_mode(numbers):
    #Mean
    total = 0
    for num in numbers:
        total += num
    mean = total / len(numbers)

    #Median
    numbers.sort()
    mid = len(numbers) // 2
    median = numbers[mid]

    #Mode
    frequency = {}
    for num in numbers:
        frequency[num] = frequency.get(num, 0) + 1

    mode = numbers[0]
    max_freq = frequency[mode]
    for num in frequency:
        if frequency[num] > max_freq:
            max_freq = frequency[num]
            mode = num

    return mean, median, mode


#Main, Set-A

import random

list_a = [2, 7, 4, 1, 3, 6]
print("Pair count:", count_pairs_with_sum_10(list_a))

range_list = [5, 3, 8, 1, 0, 4]
print("Range:", find_range_of_list(range_list))

matrix = [[1, 2], [3, 4]]
power = 2
print("Matrix Power:", matrix_power(matrix, power))

text = "hippopotamus"
char, count = highest_occurring_character(text)
print("Highest occurring character:", char, "Count:", count)

random_numbers = [random.randint(1, 10) for _ in range(25)]
mean, median, mode = mean_median_mode(random_numbers)
print("Mean:", mean, "Median:", median, "Mode:", mode)
