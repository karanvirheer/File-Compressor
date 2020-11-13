from __future__ import annotations
import pytest
from random import shuffle
from compress import *
from hypothesis import given, assume, settings
from hypothesis.strategies import binary, integers, dictionaries, text
from typing import Dict
from test_decompression import *

tree = HuffmanTree(None, HuffmanTree(None, HuffmanTree(None,
                                                       HuffmanTree(104,
                                                                   None,
                                                                   None),
                                                       HuffmanTree(101,
                                                                   None,
                                                                   None)),
                                     HuffmanTree(None,
                                                 HuffmanTree(119, None,
                                                             None),
                                                 HuffmanTree(114, None,
                                                             None))),
                   HuffmanTree(None, HuffmanTree(108, None, None),
                               HuffmanTree(None,
                                           HuffmanTree(100, None, None),
                                           HuffmanTree(111, None, None))))


# build_frequency_dict tests

def test_empty_bytes_build_frequency_dict() -> None:
    assert build_frequency_dict(b'') == {}


def test_one_byte_build_frequency_dict() -> None:
    assert build_frequency_dict(b'h') == {104: 1}


def test_general_build_frequency_dict() -> None:
    assert build_frequency_dict(b'helloworld') == {104: 1, 101: 1, 108: 3,
                                                   111: 2, 119: 1, 114: 1,
                                                   100: 1}


def test_mutation_build_frequency_dict() -> None:
    b = b'helloworld'
    build_frequency_dict(b)
    assert b == b'helloworld'


# build_huffman_tree tests

def test_empty_freq_dict_build_huffman_tree() -> None:
    assert build_huffman_tree({}) == HuffmanTree() or \
           build_huffman_tree({}) == HuffmanTree(None)


def test_one_symbol_freq_pair_build_huffman_tree() -> None:
    d = {104: 1}
    t = build_huffman_tree(d)

    assert (t.left.symbol == 104 or t.right.symbol == 104) and \
           (t.right.symbol is not None or t.left.symbol is not None)


def test_general_build_huffman_tree() -> None:
    # Dependent on your implementation
    global tree
    d = {104: 1, 101: 1, 108: 3, 111: 2, 119: 1, 114: 1, 100: 1}
    assert build_huffman_tree(d) == tree


def test_mutation_build_huffman_tree() -> None:
    d = {104: 1, 101: 1, 108: 3, 111: 2, 119: 1, 114: 1, 100: 1}
    build_huffman_tree(d)
    assert d == {104: 1, 101: 1, 108: 3, 111: 2, 119: 1, 114: 1, 100: 1}


# get_codes tests

def test_empty_tree_get_codes() -> None:
    assert get_codes(HuffmanTree()) == {}


def test_general_get_codes() -> None:
    global tree

    assert get_codes(tree) == {104: '000', 101: '001', 119: '010', 114: '011',
                               108: '10', 100: '110', 111: '111'}


def test_mutation_get_codes() -> None:
    t = HuffmanTree(None, HuffmanTree(3), HuffmanTree(5))
    c = get_codes(t)
    assert t == HuffmanTree(None, HuffmanTree(3), HuffmanTree(5)) and \
           c == get_codes(HuffmanTree(None, HuffmanTree(3), HuffmanTree(5)))


# number_nodes tests

def test_empty_number_nodes() -> None:
    # Empty case not feasible as you can't number a None node
    pass


def test_general_number_nodes() -> None:
    global tree
    number_nodes(tree)

    assert tree.number == 5 and tree.right.number == 4 and tree.left.number == 2
    assert tree.left.left.number == 0 and tree.left.right.number == 1


def test_mutation_number_nodes() -> None:
    t = HuffmanTree(None, HuffmanTree(3), HuffmanTree(5))
    number_nodes(t)
    assert t == HuffmanTree(None, HuffmanTree(3), HuffmanTree(5))


# avg_length tests

def test_empty_tree_avg_length() -> None:
    f = {1: 2, 3: 4}
    t = HuffmanTree(None, None, None)

    assert avg_length(t, f) == 0.0


def test_general_tree_avg_length() -> None:
    d = {104: 1, 101: 1, 108: 3, 111: 2, 119: 1, 114: 1, 100: 1}
    global tree

    assert avg_length(tree, d) == 2.7


def test_mutation_avg_length() -> None:
    f = {1: 2, 3: 4}
    t = HuffmanTree(None, HuffmanTree(1), HuffmanTree(3))
    avg_length(t, f)

    assert f == {1: 2, 3: 4} and t == HuffmanTree(None,
                                                  HuffmanTree(1),
                                                  HuffmanTree(3))


# compress_bytes tests

def test_empty_bytes_compress_bytes() -> None:
    # See @2296 on Piazza
    t = b''
    f = {}
    assert compress_bytes(t, f) == b''


def test_one_byte_compress_bytes() -> None:
    t = b'h'
    f = {104: '0', 108: '1'}

    assert compress_bytes(t, f) == b'\x00'


def test_general_compress_bytes() -> None:
    t = b'it is corona time'
    c = {101: '000', 116: '001', 111: '010', 115: '0110', 99: '0111',
         114: '1000', 110: '1001', 97: '1010', 109: '1011',
         105: '110', 32: '111'}

    assert compress_bytes(t, c) == b'\xc7\xe6\xee\xa1Ms\xac\x00'


def test_mutation_compress_bytes() -> None:
    t = b'it is corona time'
    c = {101: '000', 116: '001', 111: '010', 115: '0110', 99: '0111',
         114: '1000', 110: '1001', 97: '1010', 109: '1011',
         105: '110', 32: '111'}

    compress_bytes(t, c)
    assert t == b'it is corona time' and c == {101: '000', 116: '001',
                                               111: '010',
                                               115: '0110', 99: '0111',
                                               114: '1000', 110: '1001',
                                               97: '1010', 109: '1011',
                                               105: '110', 32: '111'}


# tree_to_bytes tests

def test_empty_tree_to_bytes() -> None:
    t = HuffmanTree()
    number_nodes(t)
    assert list(tree_to_bytes(t)) == []


def test_general_tree_to_bytes() -> None:
    global tree
    number_nodes(tree)
    assert list(tree_to_bytes(tree)) == [0, 104, 0, 101, 0, 119, 0, 114, 1,
                                         0, 1, 1, 0, 100, 0, 111, 0, 108,
                                         1, 3, 1, 2, 1, 4]


def test_mutation_tree_to_bytes() -> None:
    t = HuffmanTree(None, HuffmanTree(3), HuffmanTree(5))
    tree_to_bytes(t)
    assert t == HuffmanTree(None, HuffmanTree(3), HuffmanTree(5))

# Test Improve Tree


def shuffle_dictionary_values(d: Dict[int, int]) -> None:
    """Shuffle the values of d"""
    all_values = list(d.values())
    random.shuffle(all_values)
    for k in d:
        d[k] = all_values.pop()


@given(dictionaries(integers(0, 255), integers(1, 1000), dict, 2, 256))
def test_improve_tree_shuffling_dict(d: Dict[int, int]) -> None:
    """Motivation: the avg_length for a tree created with build_huffman_tree
    should be the same as the one created after shuffling the values
    of the frequency dictionary and improving the tree"""
    t = build_huffman_tree(d)
    original = avg_length(t, d)
    shuffle_dictionary_values(d)
    improve_tree(t, d)
    assert avg_length(t, d) == original, \
        f"new:={avg_length(t, d)}\toriginal:={original}"


if __name__ == '__main__':
    pytest.main(['test_a2_compression.py'])
