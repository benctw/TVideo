from typing import NamedTuple

# 2D Point
class Point2D(NamedTuple):
    x: int
    y: int

class Box(NamedTuple):
    p1: Point2D
    p2: Point2D

