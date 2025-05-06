import pytest
from fairpyx.allocate_worst_case import allocate

# דוגמה 1 
def test_example_1():
    value_matrix = [[6, 3, 1], [2, 5, 5]]
    threshold = [6, 6]
    result = allocate(value_matrix, threshold)
    assert result == {0: [0], 1: [1, 2]}

#דוגמה 2
def test_example_2():
    value_matrix = [[7, 2, 1, 1], [3, 6, 1, 2], [2, 3, 5, 5]]
    threshold = [7, 6, 9]
    result = allocate(value_matrix, threshold)
    assert result == {2: [2, 3], 0: [0], 1: [1]}

# קלט ריק
def test_empty_input():
    assert allocate([], []) == {}

# קלט לא תקני – מימדים שונים
def test_invalid_input_length():
    value_matrix = [[1, 2], [3, 4]]
    threshold = [5]  # פחות מדי ספים
    with pytest.raises(Exception):
        allocate(value_matrix, threshold)

# קלט קצה – ערכים נמוכים מאוד
def test_zero_thresholds():
    value_matrix = [[0, 0], [0, 0]]
    threshold = [0, 0]
    result = allocate(value_matrix, threshold)
    assert set(result.keys()) == {0, 1}

# קלט אקראי והשוואה לאלגוריתם נאיבי (חיפוש שלם או תנאי תקינות)
import random

def test_random_input_non_empty_allocation():
    agents = 5
    items = 10
    value_matrix = [[random.randint(1, 10) for _ in range(items)] for _ in range(agents)]
    threshold = [sum(row) / len(row) / 2 for row in value_matrix]  # חצי מממוצע הערכים לסוכן
    result = allocate(value_matrix, threshold)

    # בדיקה: כל סוכן שהוקצה לו משהו עומד בתנאי הסף
    for agent, bundle in result.items():
        value = sum(value_matrix[agent][item] for item in bundle)
        assert value >= threshold[agent]
