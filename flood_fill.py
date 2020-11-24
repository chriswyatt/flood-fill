#!/usr/bin/env python3

"""
Flood fill

A flood fill algorithm that calculates the boundaries on-the-fly, which
makes it more memory efficient in some situations
"""

from collections import namedtuple
from itertools import chain, count
from math import floor, log10

import numpy as np

# DEBUG
# from PIL import Image


# Threshold, used to calculate whether a mine is at a given position
THRESHOLD = 23


# Horizontal line
Line = namedtuple('Line', ('x', 'y', 'width'))


def _get_int_digits(value):
    """
    Get the digits of an integer, e.g. 12345 yields 1, 2, 3, 4, 5
    """

    remainder = abs(value)

    if remainder == 0:
        yield remainder
        return

    for ix in range(floor(log10(remainder)), 0, -1):
        digit, remainder = divmod(remainder, 10 ** ix)
        yield digit

    yield remainder


def _check_mine_at_position(x, y):
    """
    Returns True if the sum of all digits of X and Y are greater than
    23; otherwise False

    Examples:
    X=23, Y=34 (sum: 12) => False
    X=345, Y=789 (sum: 36) => True
    """

    digits = chain.from_iterable(map(_get_int_digits, (x, y)))
    return sum(digits) > THRESHOLD


def _slice_line(line, slice_obj):
    """
    Slice a Line tuple into another Line tuple, using a NumPy slice
    object. The NumPy slice object contains the X offsets rather than
    the absolute values of X.
    """

    assert slice_obj.step is None
    x1_offset = int(slice_obj.start)
    x2_offset = int(slice_obj.stop)
    return Line(
        x=line.x + x1_offset,
        y=line.y,
        width=x2_offset - x1_offset,
    )


def _find_safe_clumps_within_line(line):
    """
    Given a Line tuple, find all the contiguous regions of the line that
    do not contain any mines, for example:

    Given a line at (2, 3), where X represents a mine:
    [23X567X9012X]

    Returns lines:
    (
        Line(x=2, y=3, width=2),
        Line(x=5, y=3, width=3),
        Line(x=9, y=4, width=4),
    ),
    """

    # We use NumPy here, as it has built-in functionality for finding
    # contiguous regions (i.e. clumps)

    arr = np.ma.masked_array(np.arange(line.width))

    mine_offsets = [
        offset
        for offset in range(line.width)
        if _check_mine_at_position(line.x + offset, line.y) is True]

    arr[mine_offsets] = np.ma.masked
    return (_slice_line(line, s) for s in np.ma.clump_unmasked(arr))


def _x_adjust(x, y, step):
    """
    Adjust x towards the nearest clump boundary. E.g., if x is 5, step
    is -1 and the nearest mine is at x=2, then this will return 3

    Note: this depends on the supplied position not containing a mine
    """

    assert _check_mine_at_position(x, y) is False

    return next(
        x
        for x in count(start=x, step=step)
        if _check_mine_at_position(x + step, y) is True)


def _normalize_line(line):
    """
    Normalize a Line tuple. This ensures that X1 and X2 are either mines
    or are at the boundary of a safe clump.

    In other words, if X1 has no mines to the left of it, X1 is extended
    up until the point a mine is reached; likewise, if X2 has no mines
    to the right of it, X2 is extended up until the point a mine is
    reached.
    """

    x1 = line.x
    x2 = line.x + line.width
    y = line.y

    # Extend X1 to the nearest safe clump boundary
    if _check_mine_at_position(x1, y) is False:
        x1 = _x_adjust(x1, y, -1)

    # Extend X2 to the nearest safe clump boundary
    if _check_mine_at_position(x2, y) is False:
        x2 = _x_adjust(x2, y, 1) + 1

    return Line(x=x1, y=y, width=x2 - x1)


def _find_safe_clumps(line):
    """
    Given a Line tuple, find any lines above or below, touching the
    line, that are contiguous regions not containing any mines.
    """

    # Lines directly above and below the supplied line

    # The leftmost and rightmost boundaries of each line may be extended
    # (see _normalize_line())

    parallel_lines = map(
        _normalize_line,
        (
            Line(line.x, line.y + y_offset, line.width)
            for y_offset in range(-1, 3, 2)),
    )

    # Yields all clumps above and below the supplied line

    return chain.from_iterable(map(
        _find_safe_clumps_within_line,
        parallel_lines,
    ))


# DEBUG
# def _get_dimensions(clumps):
#     x1 = 0
#     x2 = 0
#     y1 = 0
#     y2 = 0
#
#     for clump in clumps:
#         x1 = min(x1, clump.x)
#         x2 = max(x2, clump.x + clump.width)
#         y1 = min(y1, clump.y)
#         y2 = max(y2, clump.y + 1)
#
#     # x_origin_offset, y_origin_offset, width, height
#     return 0 - x1, 0 - y1, x2 - x1, y2 - y1


# DEBUG
# def verify_clumps(clumps):
#     (x_origin_offset,
#      y_origin_offset,
#      width,
#      height) = _get_dimensions(clumps)
#
#     arr = np.zeros((height, width), dtype=np.ubyte)
#
#     for clump in clumps:
#         y = clump.y + y_origin_offset
#         x1 = clump.x + x_origin_offset
#         x2 = x1 + clump.width
#         arr[y, x1:x2] |= 0x1
#
#     for row in range(height):
#         y = row - y_origin_offset
#
#         for col in range(width):
#             x = col - x_origin_offset
#
#             if _check_mine_at_position(x, y) is False:
#                 arr[row, col] |= 0x2
#
#     img = Image.new("RGB", (width, height), (0, 0, 0))
#
#     for row in range(height):
#         for col in range(width):
#             value = int(arr[row, col])
#
#             img.putpixel(
#                 (col, row),
#                 (
#                     255 if (value & 0x1) else 0,
#                     0,
#                     255 if (value & 0x2) else 0,
#                 ),
#             )
#
#     img.show()


def main():
    # A clump is contiguous space between 2 mines (along the X axis).
    # This gets the first clump, centered across the origin.
    first_clump = _normalize_line(Line(x=0, y=0, width=0))

    # Clumps that have not yet been checked for neighbouring clumps
    unprocessed_clumps = {first_clump}

    # Clumps that have been checked for neighbouring clumps
    processed_clumps = set()

    # As we process clumps, the unprocessed_clumps set will continue
    # to fill until there are no clumps left. This is slightly
    # inefficient, as we will find the same clumps more than once.

    while len(unprocessed_clumps) > 0:
        # Pick any old clump from the pile
        clump = next(iter(unprocessed_clumps))

        # These are all the neighbouring clumps that are either fully or
        # partly above or below the clump
        found_clumps = frozenset(_find_safe_clumps(clump))

        # Add the clumps to the queue, removing any clumps that have
        # already been processed
        unprocessed_clumps.update(found_clumps - processed_clumps)

        # Remove this clump from the queue and mark as processed
        unprocessed_clumps.remove(clump)
        processed_clumps.add(clump)

    # DEBUG
    # verify_clumps(processed_clumps)

    print(sum(c.width for c in processed_clumps))


if __name__ == '__main__':
    main()
