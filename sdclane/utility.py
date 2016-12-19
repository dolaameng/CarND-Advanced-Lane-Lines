"""
Helper functions to build pipeline in a functional style.
"""

import cv2
import functools

## Pipeline related

def make_pipeline(steps):
    """Create a pipeline composed of each step
    """
    def compose2(f, g):
        return lambda x: g(f(x))
    return functools.reduce(compose2, steps)
def AND(f, g):
    """Return the AND operation result of two steps
    """
    def _and(x):
        return f(x) & g(x)
    return _and
def OR(f, g):
    """Return the OR operation result of two steps
    """
    def _or(x):
        return f(x) | g(x)
    return _or

## Image IO

def read_rgb_imgs(img_files):
    return [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            for f in img_files]