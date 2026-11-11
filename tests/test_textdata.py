"""data.text_dataset: any record shape, any tokenizer, packed windows, prompt labelled -100."""
import json
import os
import tempfile
import unittest

import numpy as np

from apsi import textdata as td
from apsi.stdlib import cross_entropy


class TextData(unittest.TestCase):
    def test_detect(self):
        self.assertEqual(td.detect("x"), "text")
        self.assertEqual(td.detect({"messages": []}), "chat")
        self.assertEqual(td.detect({"conversations": [{"from": "human", "value": "a"}]}), "chat")
        self.assertEqual(td.detect({"instruction": "a", "output": "b"}), "instruction")
        self.assertEqual(td.detect({"prompt": "a", "completion": "b"}), "completion")
        self.assertEqual(td.detect({"question": "a", "answer": "b"}), "qa")
        self.assertEqual(td.detect({"content": "a"}), "text")

    def test_windows_are_shifted_and_packed(self):
        ds = td.text_dataset(["abcdefghij", "klmnop"], tokenizer="bytes", seq_len=4, eos=False)
        got = list(ds)
        flat = list(b"abcdefghijklmnop")
        self.assertEqual(len(got), 3)
        for k, (x, y) in enumerate(got):
            self.assertEqual(x.tolist(), flat[4 * k:4 * k + 4])
            self.assertEqual(y.tolist(), flat[4 * k + 1:4 * k + 5])

    def test_prompt_not_trained(self):
        rec = {"prompt": "Q?", "completion": "A!"}
        (x, y), = list(td.text_dataset([rec], seq_len=8, pack=False, eos=False))
        ids = list(b"Q?A!")
        self.assertEqual(x.tolist()[:3], ids[:3])
        self.assertEqual(y.tolist()[:3], [-100, ids[2], ids[3]])     # predicting '?' is prompt, 'A' and '!' train
        self.assertTrue(all(v == -100 for v in y.tolist()[3:]))      # padding
        (x2, y2), = list(td.text_dataset([rec], seq_len=8, pack=False, eos=False, train_on_prompt=True))
        self.assertEqual(y2.tolist()[0], ids[1])

    def test_chat_formats_and_files(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.jsonl")
            with open(p, "w") as fh:
                fh.write(json.dumps({"conversations": [{"from": "human", "value": "hi"},
                                                        {"from": "gpt", "value": "yo"}]}) + "\n")
            parts = td.to_parts(next(td.read_records(p)), chat_style="chatml")
            text = "".join(t for _, t in parts)
            self.assertIn("<|im_start|>user\nhi<|im_end|>", text)
            self.assertIn("<|im_start|>assistant\nyo<|im_end|>", text)
            self.assertEqual([t for tr, t in parts if tr], ["yo<|im_end|>"])
            c = os.path.join(d, "t.csv")
            with open(c, "w") as fh:
                fh.write("q,a\nwhy,because\n")
            out = list(td.text_dataset(c, seq_len=64, template="Q: {q} A: {a}", pack=False, eos=False))
            self.assertEqual(bytes(v for v in out[0][0].tolist() if v).decode(), "Q: why A: becaus")

    def test_custom_tokenizer_function_and_template_function(self):
        ds = td.text_dataset([{"a": 1, "b": 2}], tokenizer=lambda s: [ord(c) % 7 for c in s], seq_len=3,
                             template=lambda r: f"{r['a']}+{r['b']}", eos=5)
        (x, y), = list(ds)
        self.assertEqual(x.tolist(), [ord("1") % 7, ord("+") % 7, ord("2") % 7])
        self.assertEqual(y.tolist()[-1], 5)

    def test_cross_entropy_ignores_minus_100(self):
        logits = np.log(np.array([[0.5, 0.5], [0.9, 0.1]]))
        full = float(cross_entropy(logits, np.array([0, 0])))
        part = float(cross_entropy(logits, np.array([-100, 0])))
        self.assertAlmostEqual(part, -np.log(0.9), places=6)
        self.assertNotAlmostEqual(full, part)


if __name__ == "__main__":
    unittest.main()
