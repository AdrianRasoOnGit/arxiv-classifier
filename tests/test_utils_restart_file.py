
from pathlib import Path
from arxiv_classifier.utils.restart_file import restart_file

def test_restart_file_deletes_existing(tmp_path):
    f = tmp_path / "will_be_removed.txt"
    f.write_text("x")
    assert f.exists()
    restart_file(f)
    assert not f.exists()

def test_restart_file_missing_is_ok(tmp_path):
    f = tmp_path / "missing.txt"
    # should not raise
    restart_file(f)
    assert not f.exists()
