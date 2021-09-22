#!/usr/bin/env python3

from typing import AnyStr, TypeVar
AnyType = TypeVar("T")

from core.configuration.consts import VERBOSE

class Edge:
    """
    This class represent an edge between nodes in neural networks (or graphs),
    and acts like a regular edge in other graphs
    """
    def __init__(self, src: AnyStr, dest:AnyStr, weight:float):
        # src, dest are names (ids) of nodes (strings)
        self.src = src
        self.dest = dest
        self.weight = weight

    def __eq__(self, other:AnyType, verbose:bool=VERBOSE) -> bool:
        if self.src != other.src:
            if verbose:
                print("self.src ({}) != other.src ({})".format(self.src, other.src))
            return False
        if self.dest != other.dest:
            if verbose:
                print("self.dest ({}) != other.dest ({})".format(self.dest, other.dest))
            return False
        if self.weight != other.weight:
            if verbose:
                print("self.weight ({}) != other.weight ({})".format(self.weight, other.weight))
            return False
        return True

    def __str__(self) -> str:
        return "{}--({})-->{}".format(self.src, self.weight, self.dest)
