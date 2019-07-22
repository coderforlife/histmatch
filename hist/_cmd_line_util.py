"""Utilities for the command line programs."""

from numpy import nan, inf
import hist.exact.va as va

METHODS = ('classic', 'arbitrary', 'rand', 'na', 'nv', 'gl', 'ml', 'lc', 'lm', 'wa', 'swa', 'va',
           'optimum')

def add_method_arg(parser):
    """Add the method argument to an argument parser object."""
    parser.add_argument('method', choices=METHODS, help='method of histogram equalization')

def add_kwargs_arg(parser):
    """Add the kwargs arg to an argument parser object which accepts a series of k=v arguments."""
    parser.add_argument('kwargs', type=__kwargs_arg, nargs='*', help='any special keyword '+
                        'arguments to pass to the method, formated as key=value with value being '+
                        'a valid Python literal or one of the special values nan, inf, -inf, N4, '+
                        'N8, N8_DIST, N6, N18, N18_DIST, N26, N26_DIST')

__CONVERSIONS = {
    'nan': nan,
    'inf': inf,
    '-inf': -inf,
    'N4': va.CONNECTIVITY_N4,
    'N8': va.CONNECTIVITY_N8,
    'N8_DIST': va.CONNECTIVITY_N8_DIST,
    'N6': va.CONNECTIVITY3_N6,
    'N18': va.CONNECTIVITY3_N18,
    'N18_DIST': va.CONNECTIVITY3_N18_DIST,
    'N26': va.CONNECTIVITY3_N26,
    'N26_DIST': va.CONNECTIVITY3_N26_DIST,
}

def __kwargs_arg(value):
    import ast
    key, val = value.split('=', 1)
    return key, __CONVERSIONS[val] if val in __CONVERSIONS else ast.literal_eval(val)
