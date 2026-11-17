import json
from pathlib import Path
import tempfile
import unittest

from datasets.preprocessing import (
    DOMAINS, assert_template_disjoint, normalize, render_records,
    split_records, write_jsonl,
)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
        return path

    def records(self, groups=8, group_size=2):
        return [dict(id=f"{domain}-{i}-{j}", source_id=f"{domain}-{i}", domain=domain,
                     text=f"A request with value {i} and part {j}.")
                for domain in DOMAINS.values() for i in range(groups) for j in range(group_size)]

    def test_medquad_grouping_and_explicit_annotations(self):
        path = self.write("medical.xml", '<Document url="https://example.org/item"><QAPairs>'
                          '<QAPair><Question>Question one?</Question><Answer>Answer.</Answer></QAPair>'
                          '<QAPair><Question>Question two?</Question><Answer>Answer.</Answer></QAPair>'
                          '</QAPairs></Document>')
        rows = normalize("medquad", path)
        self.assertEqual(2, len(rows))
        self.assertEqual(rows[0]["source_id"], rows[1]["source_id"])
        self.assertNotIn("intent", rows[0])
        annotated = normalize("medquad", path, {rows[0]["id"]: {"intent": "question"}})
        self.assertEqual("question", annotated[0]["intent"])
        with self.assertRaises(ValueError):
            normalize("medquad", path, {"missing": {"intent": "x"}})

    def test_finqa_preserves_table_but_not_answer_label(self):
        path = self.write("finance.json", [{"id": "report-1", "pre_text": ["Report"],
                          "table": [["Year", "Revenue"], ["2020", "10"]],
                          "post_text": ["Notes"], "qa": {"question": "What changed?",
                                                                      "answer": "secret label"}}])
        rows = normalize("finqa", path)
        self.assertIn("Year | Revenue", rows[0]["text"])
        self.assertIn("What changed?", rows[0]["text"])
        self.assertNotIn("secret label", rows[0]["text"])

    def test_cuad_contract_grouping(self):
        path = self.write("legal.json", {"data": [{"title": "contract", "paragraphs": [
            {"context": "The contract terms.", "qas": [
                {"id": "q1", "question": "First clause?"},
                {"id": "q2", "question": "Second clause?"}]}]}]})
        rows = normalize("cuad", path)
        self.assertEqual(rows[0]["source_id"], rows[1]["source_id"])
        self.assertNotEqual(rows[0]["id"], rows[1]["id"])

    def test_codexglue_same_code_keeps_source_group(self):
        path = self.write("code.jsonl", '\n'.join(json.dumps(row) for row in [
            {"idx": 1, "func": "return 1;", "text": "First request"},
            {"idx": 2, "func": "return 1;", "text": "Second request"}]))
        rows = normalize("codexglue", path)
        self.assertEqual(rows[0]["source_id"], rows[1]["source_id"])
        self.assertNotIn("keyword", rows[0])

    def test_exact_quotas_and_source_disjointness(self):
        records = self.records()
        split = split_records(records, train_per_domain=7, test_per_domain=5, seed=17)
        self.assertEqual(28, len(split["train"]))
        self.assertEqual(20, len(split["test"]))
        self.assertFalse({r["source_id"] for r in split["train"]} &
                         {r["source_id"] for r in split["test"]})
        self.assertEqual(split, split_records(reversed(records), train_per_domain=7,
                                              test_per_domain=5, seed=17))

    def test_impossible_grouped_quota_fails(self):
        with self.assertRaisesRegex(ValueError, "Source groups"):
            split_records(self.records(groups=2, group_size=5),
                          train_per_domain=6, test_per_domain=4)

    def test_unquoted_split_keeps_every_record(self):
        rows = self.records()
        split = split_records(rows)
        self.assertEqual(len(rows), len(split["train"]) + len(split["test"]))
        self.assertEqual(48, len(split["train"]))

    def test_shared_shell_and_held_out_templates(self):
        split = split_records(self.records())
        train = render_records(split["train"], template_id="shared-v1")
        test = render_records(split["test"], template_id="shared-v2")
        assert_template_disjoint(train, test)
        shells = {row["prompt"].replace(row["text"], "CONTENT") for row in train}
        self.assertEqual(1, len(shells))
        with self.assertRaisesRegex(ValueError, "templates"):
            assert_template_disjoint(train, render_records(split["test"], template_id="shared-v1"))

    def test_counts_come_from_tokenizer(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=True):
                assert add_special_tokens
                return [1, 2, 3]
        rows = self.records(groups=1, group_size=1)
        rendered = render_records(rows, tokenizer=Tokenizer(), min_tokens=3, max_tokens=3)
        self.assertTrue(all(row["token_count"] == 3 for row in rendered))
        with self.assertRaises(ValueError):
            render_records(rows, min_tokens=3)
        with self.assertRaisesRegex(ValueError, "longer"):
            render_records(rows, tokenizer=Tokenizer(), max_tokens=2)
        self.assertNotIn("token_count", render_records(rows)[0])

    def test_duplicate_record_rejected(self):
        rows = self.records(groups=1, group_size=1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            split_records(rows + [rows[0]])

    def test_output_is_not_overwritten(self):
        path = self.root / "output.jsonl"
        write_jsonl(path, self.records())
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            write_jsonl(path, [])
        self.assertEqual(original, path.read_bytes())

    def test_xml_entities_are_rejected(self):
        path = self.write("entities.xml", '<!DOCTYPE foo [<!ENTITY x SYSTEM "file:///x">]><foo/>')
        with self.assertRaisesRegex(ValueError, "entities"):
            normalize("medquad", path)


if __name__ == "__main__":
    unittest.main()
