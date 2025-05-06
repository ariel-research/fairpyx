"""
dict_key: allows you to use a dict as a key for selecting a row in a pandas dataframe.

Author: Erel Segal-Halevi
Since: 2022-05
"""

import pandas

from numbers import Number

def dict_to_rows(df: pandas.DataFrame, key:dict)->pandas.DataFrame:
    """
    Accepts a DataFrame and a dict mapping column-names to values.
    Searches for rows in which the given columns contain the given values.
    Returns a dataframe with only these rows.

    >>> df = pandas.DataFrame({'a': [1,4,1], 'b': [2,5,5], 'c':[3,6,9], 'z':[123, 456, 159]})
    >>> dict_to_rows(df, {"a":1})
       a  b  c    z
    0  1  2  3  123
    2  1  5  9  159
    >>> dict_to_rows(df, {"a":1, "b":2, "c":3})
       a  b  c    z
    0  1  2  3  123
    >>> dict_to_rows(df, {"a":1, "b":2, "c":9})
    Empty DataFrame
    Columns: [a, b, c, z]
    Index: []
    >>> dict_to_rows(df, {"c":[3,9]})
       a  b  c    z
    0  1  2  3  123
    2  1  5  9  159
    >>> dict_to_rows(df, {"c":{3,9}})
       a  b  c    z
    0  1  2  3  123
    2  1  5  9  159
    """
    for k,v in key.items():
        if isinstance(v, list) or isinstance(v,set):
            df = df[df[k].isin(v)]
        else:
            df = df[df[k]==v]
    return df

def dict_to_row(df: pandas.DataFrame, key:dict)->dict:
    """
    Accepts a DataFrame and a dict mapping column-names to values.
    Searches a row in which the given columns contain the given values.
    Returns one such row if it exists (as a dict), or None if no such row exists.

    >>> df = pandas.DataFrame({'a': [1,4,1], 'b': [2,5,5], 'c':[3,6,9], 'z':[123, 456, 159]})
    >>> dict_to_row(df, {"a":1, "b":2, "c":3})
    {'a': 1, 'b': 2, 'c': 3, 'z': 123}
    
    >>> dict_to_row(df, {"a":1, "b":2, "c":9}) is None
    True
    """
    df = dict_to_rows(df, key)
    if df.empty:
        return None
    else:
        return df.iloc[0].to_dict()

def dict_to_rows_bounds(df: pandas.DataFrame, lowerbound: dict = {}, upperbound: dict = {}):
    """
    Accepts a DataFrame and two dict mapping column-names to values.
    Searches a row in which:
    * Each of the given *numeric* columns contains *at least* the values given in lowerbound, and *at most* the values given in upperbound.
    * Each of the given *non-numeric* columns contains *exactly* the given values.
    Returns a dataframe with only these rows.

    >>> df = pandas.DataFrame({'a': [1,4,1], 'b': [2,5,5], 'c':['tt','uu','vv'], 'z':[123, 456, 159], 't':[True, False, True]})
    >>> dict_to_rows_bounds(df, upperbound={"a":1, "b":2, "c":'tt'})
       a  b   c    z     t
    0  1  2  tt  123  True
    >>> dict_to_rows_bounds(df, upperbound={"a":1, "b":3, "c":'tt'})
       a  b   c    z     t
    0  1  2  tt  123  True
    >>> dict_to_rows_bounds(df, upperbound={"a":5, "b":6, "c":'uu'})
       a  b   c    z      t
    1  4  5  uu  456  False
    >>> dict_to_rows_bounds(df, upperbound={"a":2, "b":1, "c":'tt'}).empty
    True
    >>> dict_to_rows_bounds(df, upperbound={"a":1, "b":2, "c":'xx'}).empty
    True
    >>> dict_to_rows_bounds(df, lowerbound={"a":2, "b":1, "c":'uu'})
       a  b   c    z      t
    1  4  5  uu  456  False
    >>> dict_to_rows_bounds(df, lowerbound={"t":True})
       a  b   c    z     t
    0  1  2  tt  123  True
    2  1  5  vv  159  True
    >>> dict_to_rows_bounds(df, lowerbound={"t":False})
       a  b   c    z      t
    1  4  5  uu  456  False
    """
    for k,v in upperbound.items():
        if isnumber(v):
            df = df[df[k]<=v]
        else:
            df = df[df[k]==v]
    for k,v in lowerbound.items():
        if isnumber(v):
            df = df[df[k]>=v]
        else:
            df = df[df[k]==v]
    return df


def dict_to_row_bounds(df: pandas.DataFrame, lowerbound: dict = {}, upperbound: dict = {}):
    """
    Accepts a DataFrame and two dict mapping column-names to values.
    Searches a row in which:
    * Each of the given *numeric* columns contains *at least* the values given in lowerbound, and *at most* the values given in upperbound.
    * Each of the given *non-numeric* columns contains *exactly* the given values.

    Returns one such row if it exists, or None if no such row exists.

    >>> df = pandas.DataFrame({'a': [1,4,1], 'b': [2,5,5], 'c':['tt','uu','vv'], 'z':[123, 456, 159], 't':[True, False, True]})
    >>> dict_to_row_bounds(df, upperbound={"a":1, "b":2, "c":'tt'})
    {'a': 1, 'b': 2, 'c': 'tt', 'z': 123, 't': True}
    >>> dict_to_row_bounds(df, upperbound={"a":1, "b":3, "c":'tt'})
    {'a': 1, 'b': 2, 'c': 'tt', 'z': 123, 't': True}
    >>> dict_to_row_bounds(df, upperbound={"a":5, "b":6, "c":'uu'})
    {'a': 4, 'b': 5, 'c': 'uu', 'z': 456, 't': False}
    >>> dict_to_row_bounds(df, upperbound={"a":2, "b":1, "c":'tt'}) is None
    True
    >>> dict_to_row_bounds(df, upperbound={"a":1, "b":2, "c":'xx'}) is None
    True
    >>> dict_to_row_bounds(df, lowerbound={"a":2, "b":1, "c":'uu'})
    {'a': 4, 'b': 5, 'c': 'uu', 'z': 456, 't': False}
    """
    df = dict_to_rows_bounds(df, lowerbound, upperbound)
    if df.empty:
        return None
    else:
        return df.iloc[0].to_dict()


def isnumber(v)->bool:
    return isinstance(v, Number) and not isinstance(v, bool)

if __name__=="__main__":
    import doctest
    print(doctest.testmod())

