import json
import sys

sys.path.insert(0, "tools")
import check_numbers  # noqa: E402


def test_table_numbers_must_trace_to_results(tmp_path, monkeypatch):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "summary.csv").write_text("condition,f1,params\nE0,0.95871,109744\n")
    (tmp_path / "results" / "d.json").write_text(json.dumps({"x": 0.2513}))
    (tmp_path / "doc.md").write_text(
        "| ID | F1 | Params |\n|---|---|---|\n| E0 | 0.9587 | 109,744 |\n| E9 | 0.251 | 1.5 |\n\n"
        "<!-- check_numbers: skip -->\n\n| old | 49.8 |\n|---|---|\n\nprose 0.123 is not checked\n")
    monkeypatch.chdir(tmp_path)
    toks = [t for _, t in check_numbers.table_tokens((tmp_path / "doc.md").read_text())]
    assert toks == ["0.9587", "109,744", "0.251", "1.5"]
    assert check_numbers.main(["doc.md"]) == 1      # 1.5 has no source
    (tmp_path / "doc.md").write_text("| E0 | 0.9587 | 0.25 |\n|---|---|---|\n")
    assert check_numbers.main(["doc.md"]) == 0
