from testing import assert_almost_equal
from maths import city_block_distance


def test_city_block():
    var result: Float32 = city_block_distance(
        List[Float32](1.0, 2.0, 3.0),
        List[Float32](4.0, 5.0, 6.0)
    )
    assert_almost_equal(result, 9.0)