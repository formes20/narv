#!/usr/bin/env python3

from core.configuration import consts


def debug_print(something) -> None:
    print("*"*80)
    print(something)
    print("*"*80)


def verbose_debug_print(message, verbose=consts.VERBOSE) -> None:
    if verbose:
        debug_print(message)


def embed_ipython(debug_message="", verbose=consts.VERBOSE) -> None:
    verbose_debug_print("embed using IPython.embed(), message: {}".format(debug_message), verbose)
    import IPython
    IPython.embed()