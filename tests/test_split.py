import pandas as pd
from src.training.split import stratified_split

def test_stratified_split_sizes():
    X = pd.DataFrame({"a": list(range(100))})
    y = pd.Series([0]*80 + [1]*20)
    sp = stratified_split(X, y, seed=42, test_size=0.2, valid_size=0.2)
    assert len(sp.y_train) == 60
    assert len(sp.y_valid) == 20
    assert len(sp.y_test) == 20
