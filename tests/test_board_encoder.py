import numpy as np
import pytest

from hex_ml.encoding.board_encoder import encode_board_channels


def test_encode_board_channels_shape_and_values_for_player_one():
    board = np.array([[1, 2, 0], [0, 1, 2], [2, 0, 0]], dtype=np.int8)
    encoded = encode_board_channels(board, player_to_move=1)

    assert encoded.shape == (4, 3, 3)
    assert encoded.dtype == np.float32
    np.testing.assert_array_equal(encoded[0], board == 1)
    np.testing.assert_array_equal(encoded[1], board == 2)
    np.testing.assert_array_equal(encoded[2], board == 0)
    np.testing.assert_array_equal(encoded[3], np.ones((3, 3), dtype=np.float32))


def test_encode_board_channels_swaps_current_and_opponent_for_player_two():
    board = np.array([[1, 2], [0, 2]], dtype=np.int8)
    encoded = encode_board_channels(board, player_to_move=2)

    np.testing.assert_array_equal(encoded[0], board == 2)
    np.testing.assert_array_equal(encoded[1], board == 1)
    np.testing.assert_array_equal(encoded[2], board == 0)
    np.testing.assert_array_equal(encoded[3], np.full((2, 2), 2.0, dtype=np.float32))


def test_encode_board_channels_rejects_invalid_player_to_move():
    board = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(ValueError):
        encode_board_channels(board, player_to_move=3)
