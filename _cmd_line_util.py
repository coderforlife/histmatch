"""Utilities for the command line programs."""

METHODS = ('classic', 'arbitrary', 'rand', 'na', 'nv', 'gl', 'lc', 'lm', 'wa', 'va', 'optimum')

def add_method_arg(parser):
    """Add the method argument to an argument parser object."""
    parser.add_argument('method', choices=METHODS, help='method of histogram equalization')
