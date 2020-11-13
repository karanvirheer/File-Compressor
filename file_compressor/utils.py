"""
Assignment 2 starter code
CSC148, Winter 2020
Instructors: Bogdan Simion, Michael Liut, and Paul Vrbik

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2020 Bogdan Simion, Michael Liut, Paul Vrbik, Dan Zingaro
"""
from __future__ import annotations
from typing import List

# ====================
# Helper functions for manipulating bytes


def get_bit(byte: int, bit_num: int) -> int:
    """ Return bit number <bit_num> from the right within the <byte> byte.

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte: int) -> str:
    """ Return the representation of <byte> as a string of bits.

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits: str) -> int:
    """ Return the integer number corresponding to the string of bits <bits>.
    If the string <bits> has less than 8 bits, it will be padded with zeroes
    to the right.

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


def bytes_to_int(buf: bytes) -> int:
    """ Return an integer from a given 4-byte little-endian representation <buf>

    >>> bytes_to_int(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def bytes_to_nodes(buf: bytes) -> List[ReadNode]:
    """ Return a list of ReadNodes corresponding to the bytes in <buf>.

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def int32_to_bytes(num: int) -> bytes:
    """ Return the <num> integer converted to a bytes object.
    The integer is assumed to contain a 32-bit (4-byte) number.
    Note: In Python3, ints are actually variable size and can even be larger
    than 64-bits. For our purposes though, we expect the size to be a number
    that does not exceed 4 bytes.

    >>> list(int32_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte) num
    return num.to_bytes(4, "little")


class ReadNode:
    """ A node as read from a compressed file.
    Each node consists of type and data information as described in the handout.
    This class offers a clean way to collect this information for each node.

    Public Attributes:
    ===========
    l_type: 0/1 (if the corresponding HuffmanTree's left is a leaf)
    l_data: a symbol or the node number of a HuffmanTree's left
    r_type: 0/1 (if the corresponding HuffmanTree's right is a leaf)
    r_data: a symbol or the node number of a HuffmanTree's right
    """
    l_type: int
    l_data: int
    r_type: int
    r_data: int

    def __init__(self, l_type: int, l_data: int,
                 r_type: int, r_data: int) -> None:
        """ Create a new ReadNode with the given parameters."""
        self.l_type, self.l_data = l_type, l_data
        self.r_type, self.r_data = r_type, r_data

    def __repr__(self) -> str:
        """ Return constructor-style string representation of this ReadNode."""
        return 'ReadNode({}, {}, {}, {})'.format(
            self.l_type, self.l_data, self.r_type, self.r_data)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'allowed-import-modules': [
            'python_ta', 'doctest', '__future__', 'typing'
        ]
    })
