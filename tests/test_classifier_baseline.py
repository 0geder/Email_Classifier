from classifier_baseline import BaselineClassifier
from fixtures import ALL_CATEGORIES, SYNTHETIC_TRAIN
from ingestion import Email


def _fitted():
    return BaselineClassifier().fit(SYNTHETIC_TRAIN)


def test_predicts_a_valid_category_for_clear_examples():
    model = _fitted()
    query = Email("99", "Insurance claim for flood damage", "x@x.com", "2025-02-01",
                   "I need to file an insurance claim after a flood damaged my kitchen.", "q.html")
    [pred] = model.predict([query])
    assert pred.predicted_category in ALL_CATEGORIES
    assert pred.email_id == "99"


def test_confidence_score_within_bounds():
    model = _fitted()
    queries = [
        Email(str(i), s, "x@x.com", "2025-01-01", b, "q.html")
        for i, (s, b) in enumerate(
            [
                ("", ""),
                ("a" * 5000, "b " * 5000),
                ("非常感谢您的邮件", "这是一个测试"),
                ("Re: Re: Re: fwd", "???"),
            ]
        )
    ]
    predictions = model.predict(queries)
    for p in predictions:
        assert 0.0 <= p.confidence_score <= 1.0
        assert p.predicted_category in ALL_CATEGORIES


def test_predict_before_fit_raises():
    import pytest

    model = BaselineClassifier()
    with pytest.raises(RuntimeError):
        model.predict(SYNTHETIC_TRAIN)


def test_predict_is_order_preserving():
    model = _fitted()
    preds = model.predict(SYNTHETIC_TRAIN)
    assert [p.email_id for p in preds] == [e.email_id for e in SYNTHETIC_TRAIN]
