from src.train import make_data


def test_make_data_shape():
    df = make_data(n=100)
    assert df.shape == (100, 4)


def test_price_is_positive_on_average():
    df = make_data()
    assert df["price"].mean() > 0
