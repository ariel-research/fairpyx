import pytest
import networkz as nx

class TestExample:
    def test_ex(self):
        G = nx.Graph()
        assert G.is_directed() == False
