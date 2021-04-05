# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 10:24:57 2021

@author: babay
"""


# Argmax - Øyvind look what you turned me into
# l - list to iterate over
# f - function to apply to those iterables. If not supplied argmax essentially becomes a max function
# *args - the whatever arguments that thhe f takes in in addition to the iterable element
def argmax(l, f = lambda e : e, *args):
    b = l[0]
    for e in l[1:]:
        if f(e, *args) > f(b, *args):
            b = e
    return b

# Let's you know if a container is empty or not - a lot easier to read 
def is_empty(c):
    if not c:
        return True
     
    return False