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


# Same as argmax, but looks for smallest value
def argmin(l, f = lambda e : e, *args):
    b = l[0]
    for e in l[1:]:
        if f(e, *args) < f(b, *args):
            b = e
    return b


# Let's you know if a container is empty or not - a lot easier to read 
def is_empty(c):
    if not c:
        return True
     
    return False