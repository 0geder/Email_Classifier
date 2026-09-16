import pytest

from fixtures import ALL_CATEGORIES, SYNTHETIC_TRAIN
from ingestion import Email

pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")


def _fitted():
    from classifier_embedding import EmbeddingClassifier

    try:
        return EmbeddingClassifier().fit(SYNTHETIC_TRAIN)
    except Exception as exc:  # model download can fail offline
        pytest.skip(f"embedding model unavailable: {exc}")


def test_predicts_a_valid_category_for_clear_examples():
    model = _fitted()
    query = Email("99", "Mortgage application update", "x@x.com", "2025-02-01",
                   "Could you tell me the status of the home loan I applied for?", "q.html")
    [pred] = model.predict([query])
    assert pred.predicted_category in ALL_CATEGORIES


def test_confidence_score_within_bounds():
    model = _fitted()
    queries = [
        Email(str(i), s, "x@x.com", "2025-01-01", b, "q.html")
        for i, (s, b) in enumerate(
            [
                ("", ""),
                ("a" * 5000, "b " * 5000),
                ("非常感谢您的邮件", "这是一个测试"),
            ]
        )
    ]
    predictions = model.predict(queries)
    for p in predictions:
        assert 0.0 <= p.confidence_score <= 1.0
        assert p.predicted_category in ALL_CATEGORIES


def test_predict_before_fit_raises():
    from classifier_embedding import EmbeddingClassifier

    model = EmbeddingClassifier()
    with pytest.raises(RuntimeError):
        model.predict(SYNTHETIC_TRAIN)
