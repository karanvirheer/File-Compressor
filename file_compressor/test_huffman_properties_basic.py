from __future__ import annotations
import pytest
from random import shuffle
from compress import *
from hypothesis import given, assume, settings
from hypothesis.strategies import binary, integers, dictionaries, text
from typing import Dict

settings.register_profile("norand", settings(
    derandomize=True, max_examples=200))
settings.load_profile("norand")


# === Test Byte Utilities ===
# Technically, these utilities are given to you in the starter code, so these
# first 3 tests below are just intended as a sanity check to make sure that you
# did not modify these methods and are therefore using them incorrectly.
# You will not be submitting utils.py anyway, so these first three tests are
# solely for your own benefit, as a sanity check.

@given(integers(0, 255))
def test_byte_to_bits(b: int) -> None:
    """ Test that byte_to_bits produces binary strings of length 8."""
    assert set(byte_to_bits(b)).issubset({"0", "1"})
    assert len(byte_to_bits(b)) == 8


@given(text(["0", "1"], 0, 8))
def test_bits_to_byte(s: str) -> None:
    """ Test that bits_to_byte produces a byte."""
    b = bits_to_byte(s)
    assert isinstance(b, int)
    assert 0 <= b <= 255


@given(integers(0, 255), integers(0, 7))
def test_get_bit(byte: int, bit_pos: int) -> None:
    """ Test that get_bit(byte, bit) produces  bit values."""
    b = get_bit(byte, bit_pos)
    assert isinstance(b, int)
    assert 0 <= b <= 1


# === Test the compression code ===

@given(binary(0, 1000))
def test_build_frequency_dict(byte_list: bytes) -> None:
    """ Test that build_frequency_dict returns dictionary whose values sum up
    to the number of bytes consumed.
    """
    # creates a copy of byte_list, just in case your implementation of
    # build_frequency_dict modifies the byte_list
    b, d = byte_list, build_frequency_dict(byte_list)
    assert isinstance(d, dict)
    assert sum(d.values()) == len(b)


@given(dictionaries(integers(0, 255), integers(1, 1000), dict, 2, 256))
def test_build_huffman_tree(d: Dict[int, int]) -> None:
    """ Test that build_huffman_tree returns a non-leaf HuffmanTree."""
    t = build_huffman_tree(d)
    assert isinstance(t, HuffmanTree)
    assert not t.is_leaf()


@given(dictionaries(integers(0, 255), integers(1, 1000), dict, 2, 256))
def test_get_codes(d: Dict[int, int]) -> None:
    """ Test that the sum of len(code) * freq_dict[code] is optimal, so it
    must be invariant under permutation of the dictionary.
    Note: This also tests build_huffman_tree indirectly.
    """
    t = build_huffman_tree(d)
    c1 = get_codes(t)
    d2 = list(d.items())
    shuffle(d2)
    d2 = dict(d2)
    t2 = build_huffman_tree(d2)
    c2 = get_codes(t2)
    assert sum([d[k] * len(c1[k]) for k in d]) == \
        sum([d2[k] * len(c2[k]) for k in d2])


@given(dictionaries(integers(0, 255), integers(1, 1000), dict, 2, 256))
def test_number_nodes(d: Dict[int, int]) -> None:
    """ If the root is an interior node, it must be numbered two less than the
    number of symbols, since a complete tree has one fewer interior nodes than
    it has leaves, and we are numbering from 0.
    Note: this also tests build_huffman_tree indirectly.
    """
    t = build_huffman_tree(d)
    assume(not t.is_leaf())
    count = len(d)
    number_nodes(t)
    assert count == t.number + 2


@given(dictionaries(integers(0, 255), integers(1, 1000), dict, 2, 256))
def test_avg_length(d: Dict[int, int]) -> None:
    """ Test that avg_length returns a float in the interval [0, 8], if the max
    number of symbols is 256.
    """
    t = build_huffman_tree(d)
    f = avg_length(t, d)
    assert isinstance(f, float)
    assert 0 <= f <= 8.0


@given(binary(2, 1000))
def test_compress_bytes(b: bytes) -> None:
    """ Test that compress_bytes returns a bytes object that is no longer
    than the input bytes. Also, the size of the compressed object should be
    invariant under permuting the input.
    Note: this also indirectly tests build_frequency_dict, build_huffman_tree,
    and get_codes.
    """
    d = build_frequency_dict(b)
    t = build_huffman_tree(d)
    c = get_codes(t)
    compressed = compress_bytes(b, c)
    assert isinstance(compressed, bytes)
    assert len(compressed) <= len(b)
    lst = list(b)
    new = lst.copy()
    shuffle(lst)
    b = bytes(lst)
    d = build_frequency_dict(b)
    t = build_huffman_tree(d)
    c = get_codes(t)
    compressed2 = compress_bytes(b, c)
    assert len(compressed2) == len(
        compressed), f"compressed2:={lst}\tcompressed:={new}"


@given(binary(2, 1000))
def test_tree_to_bytes(b: bytes) -> None:
    """ Test that tree_to_bytes generates a bytes representation of a postorder
    traversal of a tree's internal nodes.
    Since each internal node requires 4 bytes to represent, and there are
    1 fewer internal nodes than distinct symbols, the length of the bytes
    produced should be 4 times the length of the frequency dictionary, minus 4.
    Note: also indirectly tests build_frequency_dict, build_huffman_tree, and
    number_nodes.
    """
    d = build_frequency_dict(b)
    assume(len(d) > 1)
    t = build_huffman_tree(d)
    number_nodes(t)
    output_bytes = tree_to_bytes(t)
    dictionary_length = len(d)
    leaf_count = dictionary_length
    assert (4 * (leaf_count - 1)) == len(output_bytes)


# === Test a roundtrip conversion

@given(binary(1, 1000))
def test_round_trip_compress_bytes(b: bytes) -> None:
    """ Test that applying compress_bytes and then decompress_bytes
    will produce the original text.
    """
    text = b
    freq = build_frequency_dict(text)
    assume(len(freq) > 1)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    compressed = compress_bytes(text, codes)
    decompressed = decompress_bytes(tree, compressed, len(text))
    assert text == decompressed


if __name__ == "__main__":
    pytest.main(["test_huffman_properties_basic.py"])
