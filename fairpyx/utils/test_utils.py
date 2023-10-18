"""
Copied from dicttools library: https://github.com/trzemecki/dicttools/blob/39c19d9a5ecc965a58eed3eab18ac3e5e342fca2/dicttools/functions.py#L379C18-L379C18
"""

def stringify(d):
    """
    Returns a canonical string representation of the given dict,
    by sorting its items recursively.
    This is especially useful in doctests::

    >>> stringify({"a":1,"b":2,"c":{"d":3,"e":4}})
    '{a:1, b:2, c:{d:3, e:4}}'
    """
    d2 = {}

    for k, v in d.items():
        d2[k] = stringify(v) if isinstance(v, dict) else v

    return "{" + ", ".join(["{}:{}".format(k, v) for k, v in sorted(d2.items())]) + "}"


if __name__ == "__main__":
    import doctest, sys
    print("\n",doctest.testmod(), "\n")
