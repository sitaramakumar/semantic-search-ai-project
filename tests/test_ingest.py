import json
import os

import ingest


def test_load_manifest_returns_empty_dict_when_no_manifest_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "DOCS_DIR", str(tmp_path))
    assert ingest.load_manifest() == {}


def test_load_manifest_reads_allowed_roles_per_document(tmp_path, monkeypatch):
    manifest = {"secret.txt": {"allowed_roles": ["ADMIN"]}}
    (tmp_path / ingest.MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(ingest, "DOCS_DIR", str(tmp_path))

    loaded = ingest.load_manifest()

    assert loaded["secret.txt"]["allowed_roles"] == ["ADMIN"]


def test_a_document_missing_from_the_manifest_defaults_to_fully_public():
    # Fail-open, same choice as DocumentAccessGuard for a document with no
    # ACL record: an untagged document must not be silently locked out.
    manifest = {}
    allowed_roles = manifest.get("untagged.txt", {}).get("allowed_roles", ingest.DEFAULT_ALLOWED_ROLES)
    assert allowed_roles == ["READ", "WRITE", "ADMIN"]
