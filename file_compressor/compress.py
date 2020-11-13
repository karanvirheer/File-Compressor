from __future__ import annotations
import time
from typing import Dict, Tuple, Union, Any
from utils import *
from huffman import HuffmanTree


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> Dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freq_dic = {}
    for elem in text:
        if elem not in freq_dic:
            freq_dic[elem] = 1
        else:
            freq_dic[elem] += 1
    return freq_dic


def build_huffman_tree(freq_dict: Dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    >>> freq = {2: 6, 3: 4, 7: 5, 8: 4}
    >>> tree = build_huffman_tree(freq)
    >>> result =  HuffmanTree(None, HuffmanTree(None, \
    HuffmanTree(3, None, None), HuffmanTree(8, None, None)), \
    HuffmanTree(None, HuffmanTree(7, None, None), HuffmanTree(2, None, None)))
    >>> tree == result
    True
    >>> freq = {3: 1}
    >>> tree = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3, None, None), HuffmanTree(2, \
    None, None))
    >>> tree == result
    True
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
        small_freq = min(freq_list)
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


def get_codes(tree: HuffmanTree) -> Dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> tree = HuffmanTree(None, None, None)
    >>> d = get_codes(tree)
    >>> d == {}
    True
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> d_test = get_codes(tree)
    >>> d_test == {3: "00", 2: "01", 9: "1"}
    True
    >>> left_ext = HuffmanTree(None, HuffmanTree(2), HuffmanTree(3))
    >>> left = HuffmanTree(None, HuffmanTree(1), left_ext)
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> d = get_codes(tree)
    >>> d == {1: '00', 2: '010', 3: '011', 9: '10', 10: '11'}
    True
    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d_text = get_codes(tree)
    >>> d_text
    {3: '0', 2: '1'}
    """
    # Edge Case
    if tree is None or (tree.symbol is None and tree.is_leaf()):
        return {}
    else:
        return _get_codes_helper(tree, "")


def _get_codes_helper(tree: HuffmanTree, code: str,
                      symbol_dict: Any = None) -> Dict[int, str]:
    """
    A helper function for <get_codes> that returns a dictionary which maps
    symbols from the Huffman tree <tree> to codes <code>, and    outputs
    a dictionary <symbol_dict>.
    """

    if tree.is_leaf():
        symbol_dict[tree.symbol] = code
        return symbol_dict

    else:
        if symbol_dict is None:
            symbol_dict = {}

        symbol_dict = _get_codes_helper(tree.left, code + "0", symbol_dict)
        symbol_dict = _get_codes_helper(tree.right, code + "1", symbol_dict)

    return symbol_dict


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    >>> right_ext = HuffmanTree(None, HuffmanTree(2), HuffmanTree(3))
    >>> left = HuffmanTree(None, HuffmanTree(1), right_ext)
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.right.number
    0
    >>> tree.left.number
    1
    >>> tree.right.number
    2
    >>> tree.number
    3
    >>> left_ext = HuffmanTree(None, HuffmanTree(2), HuffmanTree(3))
    >>> left = HuffmanTree(None, left_ext, HuffmanTree(1))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.left.number
    0
    >>> tree.left.number
    1
    >>> tree.right.number
    2
    >>> tree.number
    3
    >>> tree = HuffmanTree(None)
    >>> number_nodes(tree)
    >>> tree.number

    """
    if tree.symbol is None and tree.is_leaf():
        return None
    else:
        _number_nodes_helper(tree, 0)
        return None


def _number_nodes_helper(tree: HuffmanTree, number: int = 0) -> int:
    """
    A helper function that uses a <tree> to <number> the internal nodes.
    """
    if tree.is_leaf():
        return number - 1
    else:
        number = _number_nodes_helper(tree.left, number) + 1
        number = _number_nodes_helper(tree.right, number) + 1
        tree.number = number
        return number


def _post_order_set_none(tree: HuffmanTree) -> None:
    """"
    Sets all the internal nodes of a <tree> to None.

    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> _print_postorder(tree)

    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> _print_postorder(tree)
    """

    if tree:
        _post_order_set_none(tree.left)
        _post_order_set_none(tree.right)
        tree.number = None


def avg_length(tree: HuffmanTree, freq_dict: Dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    >>> freq = build_frequency_dict(b'helloworld')
    >>> tree = build_huffman_tree(freq)
    >>> avg_length(tree, freq)
    2.7
    >>> freq = {}
    >>> tree = HuffmanTree(None, None, None)
    >>> avg_length(tree, freq)
    0
    """
    code_dict = get_codes(tree)

    freq_total, total = 0, 0

    for symbol in code_dict:
        freq_total += freq_dict[symbol] * len(code_dict[symbol])
        total += freq_dict[symbol]

    if total == 0:
        return 0

    return freq_total / total


def compress_bytes(text: bytes, codes: Dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']

    >>> a = compress_bytes(b'helloworld', \
    get_codes(build_huffman_tree(build_frequency_dict(b'helloworld'))))
    >>> [byte_to_bits(byte) for byte in a]
    ['00000110', '10111010', '11101110', '11000000']
    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    []
    """

    if not text:
        return bytes([])
    else:
        bit = ""
        lst = []
        for symbols in text:
            bit += codes[symbols]

            if len(bit) == 8:
                lst.append(bits_to_byte(bit))
                bit = ""

            elif len(bit) > 8:
                lst.append(bits_to_byte(bit[:8]))
                bit = bit[8:]

        if 0 < len(bit) < 8:
            byte = bits_to_byte(bit)
            lst.append(byte)

    return bytes(lst)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree)) #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    >>> tree = HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None),\
    HuffmanTree(12, None, None)), \
    HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 10, 0, 12, 0, 5, 0, 7, 1, 0, 1, 1]
    """
    if tree.is_leaf() and tree.symbol is None:
        return bytes([])
    else:
        return bytes(_traverse_post_order(tree))


def _traverse_post_order(tree: HuffmanTree, byte_list: List[int] = None) \
        -> List:
    """
    Traverses a <tree> in post order and appends 1 if it is a leaf,
    0 if it is a node, and other specifications.
    """
    if not tree.is_leaf():
        if byte_list is None:
            byte_list = []

        byte_list = _traverse_post_order(tree.left, byte_list)
        byte_list = _traverse_post_order(tree.right, byte_list)

        if not tree.left.is_leaf():
            byte_list.append(1)
            byte_list.append(tree.left.number)

        if tree.left.is_leaf():
            byte_list.append(0)
            byte_list.append(tree.left.symbol)

        if not tree.right.is_leaf():
            byte_list.append(1)
            byte_list.append(tree.right.number)

        if tree.right.is_leaf():
            byte_list.append(0)
            byte_list.append(tree.right.symbol)

    return byte_list


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree) +
              int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: List[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> tree = generate_tree_general(lst, 2)
    >>> result = HuffmanTree(None,HuffmanTree(None, \
    ...         HuffmanTree(10, None, None),\
    ...         HuffmanTree(12, None, None)), HuffmanTree(None,\
    ...         HuffmanTree(5, None, None),\
    ...         HuffmanTree(7, None, None)))
    >>> result == tree
    True

    >>> lst = [ReadNode(0, 104, 0, 101), ReadNode(0, 119, 0, 114), \
    ReadNode(1, 0, 1, 1), ReadNode(0, 100, 0, 111), ReadNode(0, 108, 1, 3), \
    ReadNode(1, 2, 1, 4)]
    >>> generate_tree_general(lst, len(lst)-1)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, \
    HuffmanTree(104, None, None), HuffmanTree(101, None, None)), \
    HuffmanTree(None, HuffmanTree(119, None, None), \
    HuffmanTree(114, None, None))), \
    HuffmanTree(None, HuffmanTree(108, None, None), \
    HuffmanTree(None, HuffmanTree(100, None, None), \
    HuffmanTree(111, None, None))))
    >>> lst = [ReadNode(1, 1, 1, 2), ReadNode(0, 5, 0, 7),
    ReadNode(0, 10, 0, 12)]
    >>> tree = generate_tree_general(lst, 0)
    >>> number_nodes(tree)
    >>> bytes_to_nodes(tree_to_bytes(tree))
    [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), ReadNode(1, 0, 1, 1)]
    """
    tree = HuffmanTree(None)
    tree.left = _gen_tree_helper(node_lst, root_index, True)
    tree.right = _gen_tree_helper(node_lst, root_index, False)
    return tree


# True = Left | False = Right
def _gen_tree_helper(node_lst: List[ReadNode],
                     root_index: int, flag: bool = True) -> HuffmanTree:
    """
    A helper function that generates a tree based on <node_lst> ReadNodes,
    and uses <root_index> and <flag> to do so.
    """
    # if internal node
    if node_lst[root_index].l_type == 1 and flag:

        # Making Tree
        tree = HuffmanTree(None)
        tree.number = node_lst[root_index].l_data

        # Creating Left and Right Trees
        tree.left = _gen_tree_helper(node_lst, tree.number, True)
        tree.right = _gen_tree_helper(node_lst, tree.number, False)

        return tree

    elif node_lst[root_index].r_type == 1 and not flag:
        # Making Tree
        tree = HuffmanTree(None)
        tree.number = node_lst[root_index].r_data

        # Creating Left and Right Trees
        tree.left = _gen_tree_helper(node_lst, tree.number, True)
        tree.right = _gen_tree_helper(node_lst, tree.number, False)

        return tree

    elif node_lst[root_index].l_type == 0 and flag:
        return HuffmanTree(node_lst[root_index].l_data)

    elif node_lst[root_index].r_type == 0 and not flag:
        return HuffmanTree(node_lst[root_index].r_data)

    return HuffmanTree(None)


def _find_height(tree: HuffmanTree, count: List = None) -> int:
    """
    Returns the height <count> of a <tree>.

    >>> tree = HuffmanTree(None, HuffmanTree(None, \
    HuffmanTree(None, HuffmanTree(104, None, None), \
    HuffmanTree(101, None, None)), HuffmanTree(None, \
    HuffmanTree(119, None, None), HuffmanTree(114, None, None))), \
    HuffmanTree(None, HuffmanTree(108, None, None), \
    HuffmanTree(None, HuffmanTree(100, None, None), \
    HuffmanTree(111, None, None))))
    >>> print(len(_find_height(tree)))
    6
    """
    if not tree.is_leaf():
        if count is None:
            count = []

        _find_height(tree.left, count)
        _find_height(tree.right, count)
        count.append(tree.symbol)

    return count


def generate_tree_postorder(node_lst: List[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
    HuffmanTree(7, None, None)), HuffmanTree(None, HuffmanTree(10, None,
    None),\
    HuffmanTree(12, None, None)))

    >>> lst = [ReadNode(0, 104, 0, 101), ReadNode(0, 119, 0, 114), \
    ReadNode(1, 0, 1, 1), ReadNode(0, 100, 0, 111), ReadNode(0, 108, 1, 3), \
    ReadNode(1, 2, 1, 4)]
    >>> tree = generate_tree_postorder(lst, len(lst)-1)
    >>> print(tree)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(None,
    HuffmanTree(104, None, None), HuffmanTree(101, None, None)), \
    HuffmanTree(None, HuffmanTree(119, None, None), \
    HuffmanTree(114, None, None))), \
    HuffmanTree(None, HuffmanTree(108, None, None), \
    HuffmanTree(None, HuffmanTree(100, None, None), \
    HuffmanTree(111, None, None))))
    >>> number_nodes(tree)
    >>> t = bytes_to_nodes(tree_to_bytes(tree))
    >>> t
    [ReadNode(0, 104, 0, 101), ReadNode(0, 119, 0, 114),
    ReadNode(1, 0, 1, 1),\
    ReadNode(0, 100, 0, 111), ReadNode(0, 108, 1, 3), ReadNode(1, 2, 1, 4)]

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> tree = generate_tree_postorder(lst, 2)
    >>> number_nodes(tree)
    >>> t = bytes_to_nodes(tree_to_bytes(tree))
    >>> t
    [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), ReadNode(1, 0, 1, 1)]
    """

    tree = HuffmanTree(None)
    tree.right = _post_order_helper(node_lst, root_index, False)

    right_index = _find_height(tree.right)

    if right_index is None:
        right_index = 0
    else:
        right_index = len(right_index)

    tree.left = _post_order_helper(node_lst, root_index, True, right_index)

    _post_order_set_none(tree)

    return tree


# True = Left | False = Right
def _post_order_helper(node_lst: List[ReadNode],
                       root_index: int, flag: bool = True,
                       right_index: int = 0) -> HuffmanTree:
    """
    A helper function that generates a tree based on <node_lst> ReadNodes,
    and uses <root_index> and <flag> and <right_index> to do so.
    """
    # if internal node
    if node_lst[root_index].l_type == 1 and flag:

        # Making Tree
        tree = HuffmanTree(None)

        tree.number = root_index - 1 - right_index

        # Creating Left and Right Trees
        tree.right = _post_order_helper(node_lst, tree.number, False)

        right_index = _find_height(tree.right)

        if right_index is None:
            right_index = 0
        else:
            right_index = len(right_index)

        tree.left = _post_order_helper(
            node_lst, tree.number, True, right_index)

        return tree

    elif node_lst[root_index].r_type == 1 and not flag:
        # Making Tree
        tree = HuffmanTree(None)
        tree.number = root_index - 1

        # Creating Left and Right Trees
        tree.right = _post_order_helper(node_lst, tree.number, False)

        right_index = _find_height(tree.right)

        if right_index is None:
            right_index = 0
        else:
            right_index = len(right_index)

        tree.left = _post_order_helper(
            node_lst, tree.number, True, right_index)

        return tree

    elif node_lst[root_index].l_type == 0 and flag:
        return HuffmanTree(node_lst[root_index].l_data)

    elif node_lst[root_index].r_type == 0 and not flag:
        return HuffmanTree(node_lst[root_index].r_data)

    return HuffmanTree(None)


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'

    >>> tree = build_huffman_tree(build_frequency_dict\
    (b"This is just a random sentence."))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b"This is just a random sentence.", \
             get_codes(tree)), len(b"This is just a random sentence."))
    b'This is just a random sentence.'
    """

    code_dict = get_codes(tree)
    inv_code_dict = {v: k for k, v in code_dict.items()}

    symbol = ""
    ans = []

    code = ''.join([byte_to_bits(byte) for byte in text])

    for char in code:
        symbol += char
        if symbol in inv_code_dict and len(ans) != size:
            ans.append(inv_code_dict[symbol])
            symbol = ""

    return bytes(ans)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_postorder(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: Dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    >>> left = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(97, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(98, None, None), \
    HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.36
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    >>> freq = build_frequency_dict(b'helloworld')
    >>> tree = build_huffman_tree(freq)
    >>> avg_length(tree, freq)
    2.7
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.7
    """
    code_dict = get_codes(tree)

    inv_code_dict = {v: k for k, v in code_dict.items()}
    inv_freq_dict = {v: k for k, v in freq_dict.items()}

    swap = {}

    for i in range(len(freq_dict)):

        min_freq = i  # Just to use i for something

        min_freq = min(inv_freq_dict)
        min_code = code_dict[inv_freq_dict[min_freq]]
        min_symbol = inv_freq_dict[min_freq]

        max_code = max(inv_code_dict)
        max_freq = freq_dict[inv_code_dict[max_code]]
        max_symbol = inv_code_dict[max_code]

        if min_freq < max_freq and len(min_code) < len(max_code):
            swap[min_symbol] = max_symbol
            swap[max_symbol] = min_symbol

            # Remove min freq symbol from dict
            inv_freq_dict.pop(min_freq)

            # Remove max code symbol from dict
            inv_code_dict.pop(max_code)
            inv_code_dict[min_code] = max_symbol
            code_dict[max_symbol] = min_code

            _swap(tree, swap)

            swap = {}


def _swap(tree: HuffmanTree, dic: Dict) -> None:
    """
    Swaps the values.
    """
    if tree.symbol:
        if tree.symbol in dic:
            tree.symbol = dic[tree.symbol]
    else:
        _swap(tree.left, dic)
        _swap(tree.right, dic)


if __name__ == "__main__":

    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print("Compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print("Decompressed {} in {} seconds."
              .format(fname, time.time() - start))

    # print("================ BEGINNING COMPRESSION ================")
    # file_list = ['bensound-buddy.mp3', 'bensound-buddy.wav',
    #              'bensound-sunny.mp3', 'bensound-sunny.wav',
    #              'Homer-Iliad.txt',
    #              'JulesVerne-MysteriousIsland.txt', 'julia_set.bmp',
    #              'mandelbrot_set.bmp', 'parrot.bmp', 'parrot.jpg']
    # for fname in file_list:
    #     start = time.time()
    #     compress_file('files/' + fname, 'files/' + fname + ".huf")
    #     print("Compressed {} in {} seconds."
    #           .format(fname, time.time() - start))
    #     print('------------------------------------------------')
    #
    # print("================ BEGINNING DECOMPRESSION ================")
    # for fname in file_list:
    #     start = time.time()
    #     decompress_file('files/' + fname + '.huf', 'files/' + fname + ".orig")
    #     print("Decompressed {} in {} seconds."
    #           .format(fname, time.time() - start))
    #     print('------------------------------------------------')
