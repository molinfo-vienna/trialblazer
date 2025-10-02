from trialblazer.Nerdd import TrialblazerModel


def test_predict():
    model = TrialblazerModel()
    df = model.predict(
        [
            "CN1CCCC(C1)CN2C3=CC=CC=C3SC4=CC=CC=C42 Pecazine",
        ],
    )

    row = df.iloc[0]

    # make sure that molecule names show up
    assert row["name"] == "Pecazine"

    # check that values are not null
    assert row["prediction"] == "toxic"

    # problem list should be empty
    assert len(row["problems"]) == 0
