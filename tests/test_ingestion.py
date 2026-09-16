from pathlib import Path

import pytest

from ingestion import load_labels, load_test_set, load_train_set, parse_email_file

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_parse_real_train_email_extracts_fields():
    email = parse_email_file(DATA_DIR / "train" / "email_1.html")
    assert email.email_id.isdigit()
    assert email.subject == "Account Transfer Request"
    assert "@" in email.sender
    assert email.date_received.count("-") == 2
    assert "transfer my account" in email.body.lower()


def test_load_train_set_matches_label_count():
    emails = load_train_set(DATA_DIR / "train", DATA_DIR / "train_labels.csv")
    assert len(emails) == 44
    assert all(e.true_category for e in emails)


def test_internal_email_id_is_globally_unique_across_train_and_test():
    train = load_train_set(DATA_DIR / "train", DATA_DIR / "train_labels.csv")
    test = load_test_set(DATA_DIR / "test")
    ids = [e.email_id for e in train] + [e.email_id for e in test]
    assert len(ids) == len(set(ids)), "email_id must not collide between train and test"


def test_load_train_set_raises_on_missing_label(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "email_1.html").write_text(
        '<div data-field="email_id">1</div><div class="email-body">hi</div>', encoding="utf-8"
    )
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("filename,true_category\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_train_set(tmp_path / "train", labels_csv)


def test_email_text_property_weights_subject():
    emails = load_train_set(DATA_DIR / "train", DATA_DIR / "train_labels.csv")
    e = emails[0]
    assert e.text.count(e.subject) == 2
