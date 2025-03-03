import math

def city_block_distance(x: List[Float32], y: List[Float32]) -> Float32:
    debug_assert(len(x) == len(y), "Input vectors must have the same length")
    var sum: Float32 = 0.0
    for i in range(len(x)):
        sum += abs(x[i] - y[i])

    return sum

