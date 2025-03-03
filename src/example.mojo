from maths import city_block_distance

def main():
    var result: Float32 = city_block_distance(
        List[Float32](1.0, 2.0, 3.0),
        List[Float32](4.0, 5.0, 6.0)
    )
    print(result)