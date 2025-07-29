def isnumeric(x):
    try:
        float(x)
        return True
    except TypeError:
        return False
