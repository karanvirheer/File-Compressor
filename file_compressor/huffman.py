from __future__ import annotations
from typing import Optional, Any


class HuffmanTree:
    """ A Huffman tree.
    Each Huffman tree may have a left and/or a right subtree.
    Symbols occur only at leaves.
    Each Huffman tree node has a number attribute that can be used for
    node-numbering.

    Public Attributes:
    ===========
    symbol: symbol located in this Huffman tree node, if any
    number: the number of this Huffman tree node
    left: left subtree of this Huffman tree
    right: right subtree of this Huffman tree
    """
    symbol: int
    number: int
    left: Optional[HuffmanTree]
    right: Optional[HuffmanTree]

    def __init__(self, symbol: Optional[int] = None,
                 left: Optional[HuffmanTree] = None,
                 right: Optional[HuffmanTree] = None) -> None:
        """ Create a new Huffman tree with the given parameters."""
        self.symbol = symbol
        self.left, self.right = left, right
        self.number = None

    def __eq__(self, other: Any) -> bool:
        """ Return True iff this HuffmanTree is equivalent to <other>, or False
        otherwise.

        >>> a = HuffmanTree(4)
        >>> b = HuffmanTree(4)
        >>> a == b
        True
        >>> b = HuffmanTree(5)
        >>> a == b
        False
        """
        return (isinstance(self, type(other)) and
                self.symbol == other.symbol and
                self.left == other.left and self.right == other.right)

    def __lt__(self, other: Any) -> bool:
        """ Return True iff this HuffmanTree is less than <other>."""
        return False  # arbitrarily say that one node is never less than another

    def __repr__(self) -> str:
        """ Return constructor-style string representation of this HuffmanTree.
        """
        return 'HuffmanTree({}, {}, {})'.format(self.symbol,
                                                self.left, self.right)

    def is_leaf(self) -> bool:
        """ Return True iff this HuffmanTree is a leaf, otherwise False.

        >>> t = HuffmanTree(None)
        >>> t.is_leaf()
        True
        """
        return not self.left and not self.right

    def num_nodes_to_bytes(self) -> bytes:
        """ Return the number of nodes required to represent this Huffman tree

        Precondition: this Huffman tree is already numbered.
        """
        return bytes([self.number + 1])


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'allowed-import-modules': [
            'python_ta', 'doctest', '__future__', 'typing'
        ]
    })
