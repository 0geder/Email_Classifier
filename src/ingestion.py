"""Parse the RedRock sample emails (HTML files) into structured records.

Each source file wraps a rendered "email client view" around a nested,
independently-parseable HTML fragment that is the actual message body. We
pull the metadata out of the `data-field` markers and the message text out
of the nested body, since the outer wrapper (title, styling, envelope) is
not part of the email content itself.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

DATA_FIELD_RE = re.compile(r'data-field="(?P<field>[a-z_]+)">(?P<value>[^<]*)')


@dataclass
class Email:
    email_id: str
    subject: str
    sender: str
    date_received: str
    body: str
    source_file: str
    true_category: str | None = None

    @property
    def text(self) -> str:
        """Combined text used as model input: subject carries strong signal, so
        repeat it once more than the body to weight it slightly higher for
        sparse (TF-IDF) representations; dense encoders are largely insensitive
        to this repetition."""
        return f"{self.subject}\n{self.subject}\n{self.body}"


def _extract_metadata(soup: BeautifulSoup) -> dict[str, str]:
    meta = {}
    for div in soup.select("div[data-field]"):
        meta[div["data-field"]] = div.get_text(strip=True)
    return meta


def _extract_body(soup: BeautifulSoup) -> str:
    container = soup.select_one(".email-body")
    if container is None:
        return soup.get_text(separator=" ", strip=True)

    nested_body = container.find("body")
    target = nested_body if nested_body is not None else container
    return target.get_text(separator=" ", strip=True)


def parse_email_file(path: Path) -> Email:
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    meta = _extract_metadata(soup)
    body = _extract_body(soup)

    return Email(
        email_id=meta.get("email_id", path.stem),
        subject=meta.get("subject", ""),
        sender=meta.get("sender", ""),
        date_received=meta.get("date_received", ""),
        body=body,
        source_file=path.name,
    )


def load_labels(labels_csv: Path) -> dict[str, str]:
    """filename -> true_category, keyed on the file's on-disk name (the CSV's
    own key), independent of the internal email_id field."""
    with labels_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["filename"]: row["true_category"] for row in reader}


def load_train_set(train_dir: Path, labels_csv: Path) -> list[Email]:
    labels = load_labels(labels_csv)
    emails = []
    for path in sorted(train_dir.glob("*.html")):
        email = parse_email_file(path)
        email.true_category = labels.get(path.name)
        if email.true_category is None:
            raise ValueError(f"No label found for {path.name} in {labels_csv}")
        emails.append(email)
    return emails


def load_test_set(test_dir: Path) -> list[Email]:
    return [parse_email_file(path) for path in sorted(test_dir.glob("*.html"))]
