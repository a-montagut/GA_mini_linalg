from my_package.core import ComputingEngine

def test_engine():
    engine = ComputingEngine()
    assert engine.process([1, 2, 3]) == [2, 4, 6]
