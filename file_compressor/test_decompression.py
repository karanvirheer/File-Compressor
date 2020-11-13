from __future__ import annotations
import pytest
import random
from compress import *
from hypothesis import given, assume, settings
from hypothesis.strategies import binary, integers, dictionaries, text
from typing import Dict


def generate_read_node(length):
    """
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> gen_read_node(tree)  #doctest: +NORMALIZE_WHITESPACE

    >>> tree = build_huffman_tree(build_frequency_dict(b"This is just a random sentence."))
    >>> number_nodes(tree)
    >>> generate_read_node(tree)

    >>> generate_read_node(24)
    """
    lst = [random.randrange(1, length) for i in range(length)]
    ans = []
    for i in range(0, len(lst), 4):
        o = lst[i:i+4]
        node = ReadNode(o[0], o[1], o[2], o[3])
        ans.append(node)
    return ans


def generate_random_dict(n, length) -> Dict[int, int]:
    """
    """
    my_dict = {}

    values = [random.randrange(1, n) for i in range(n)]

    for i in range(random.randrange(0, length)):

        my_dict[values[random.randrange(1, length)]] = values[i]

    return my_dict


def generate_huffman_tree(freq_dict: Dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4, 7: 5, 8: 4}
    >>> tree = build_huffman_tree(freq)
    >>> result =  HuffmanTree(None, HuffmanTree(None, \
    HuffmanTree(3, None, None), HuffmanTree(8, None, None)), \
    HuffmanTree(None, HuffmanTree(7, None, None), HuffmanTree(2, None, None)))
    >>> str_tree = str(tree)
    >>> tree_lst = str_tree.split(',')
    >>> random.shuffle(tree_lst)
    >>> new_tree = ''.join(tree_lst)
    >>> print(new_tree)
    """
    # Empty dictionary
    if freq_dict == {}:
        return HuffmanTree(None)

    # Only one item in freq_dict
    elif len(freq_dict) == 1:
        return HuffmanTree(None, HuffmanTree(list(freq_dict)[0]),
                           HuffmanTree(2))

    else:
        symbol_list = list(freq_dict)
        freq_list = list(freq_dict.values())
        while len(symbol_list) > 1:
            symbol_list, freq_list = _get_min(symbol_list, freq_list)
        return symbol_list[0]


def _get_min(symbol_list: List[Union[int, HuffmanTree]], freq_list: List[int]) \
        -> Tuple[List[Union[int, HuffmanTree]], List[int]]:
    """
    A helper function for build_huffman_tree that uses the <symbol_list> and
    <freq_list> to map the symbols and frequencies.
    """
    output = []
    total = 0
    for i in range(2):
        # Pop smallest frequency and get its index
        small_freq = random.choice(freq_list)
        index = freq_list.index(small_freq)

        # Add up the sum
        total += small_freq + i * 0

        # Get the symbol for the freq. and pop the freq from the list
        symbol = symbol_list.pop(index)
        freq_list.pop(index)

        if isinstance(symbol, int):
            output.append(HuffmanTree(symbol))
        elif isinstance(symbol, HuffmanTree):
            output.append(symbol)

    tree = HuffmanTree(None, output[0], output[1])

    symbol_list.append(tree)
    freq_list.append(total)
    return symbol_list, freq_list

