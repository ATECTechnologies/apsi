"""tokens.focus / unfold / fit_chat / outline: within budget, the right parts kept, nothing lost for good."""
import inspect
import os
import unittest

from apsi import token_focus as F, tokens as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CODE = '''"""A tiny shop."""
import math

TAX = 0.07


def price_with_tax(price):
    """The price a customer pays."""
    return round(price * (1 + TAX), 2)


def shipping(weight):
    """Shipping by weight in kilograms."""
    if weight < 1:
        return 4.0
    return 4.0 + math.ceil(weight - 1) * 1.5


def checkout(cart):
    """The total of a cart: every item with tax, plus shipping."""
    total = sum(price_with_tax(item["price"]) for item in cart)
    return total + shipping(sum(item["kg"] for item in cart))


def render_banner(text):
    """An ASCII banner for the shop front."""
    line = "*" * (len(text) + 4)
    return f"{line}\\n* {text} *\\n{line}"


def colour_palette():
    """The colours of the shop's website."""
    return ["#112233", "#445566", "#778899", "#aabbcc", "#ddeeff"] * 20
'''


class Focus(unittest.TestCase):
    def test_python_keeps_the_function_and_what_it_calls(self):
        f = F.focus(CODE, "why is the checkout total wrong?", budget=150)
        self.assertLessEqual(f.tokens, 150)
        kept = " ".join(f.units[i].name for i in f.kept)
        self.assertIn("def checkout", kept)
        self.assertNotIn("def colour_palette", kept)
        self.assertIn("[[omitted", f.text)
        self.assertEqual(F.unfold(f), CODE)                       # every part can come back, exactly

    def test_a_context_that_fits_is_untouched(self):
        f = F.focus(CODE, "anything", budget=10_000)
        self.assertEqual(f.text, CODE)
        self.assertEqual(f.saved.after, f.saved.before)

    def test_large_sources_stay_within_budget_and_unfold_exactly(self):
        for path, q in (("apsi/lexer.py", "how are string escapes handled"),
                        ("apsi/gla.py", "how is the router balance loss computed"),
                        ("docs/GPU.md", "how do I lower the temperature of the card")):
            text = open(os.path.join(ROOT, path), encoding="utf-8").read()
            for budget in (400, 1500):
                f = F.focus(text, q, budget=budget)
                self.assertLessEqual(f.tokens, budget, (path, budget))
                self.assertEqual(F.unfold(f), text, (path, budget))
                self.assertGreater(len(f.kept) + len(f.outlined), 0, (path, budget))

    def test_text_picks_the_relevant_section(self):
        text = open(os.path.join(ROOT, "docs/GPU.md"), encoding="utf-8").read()
        f = F.focus(text, "the card is hot or loud: how do I lower its temperature", budget=600)
        self.assertIn("hot or loud", f.text)

    def test_long_function_keeps_only_the_part_that_matters(self):
        import apsi.lexer as lexer
        src = inspect.getsource(lexer)
        f = F.focus(src, "string escapes like \\x41 and \\u00e9", budget=1500)
        self.assertIn("_HEX_ESC", f.text)
        self.assertLessEqual(f.tokens, 1500)

    def test_unfold_selected_blocks(self):
        f = F.focus(CODE, "checkout total", budget=150)
        first = min(f.blocks)
        some = F.unfold(f, [first])
        self.assertIn(f.blocks[first][1], some)
        self.assertNotIn(f.blocks[first][0], some)

    def test_outline(self):
        out = F.outline(CODE)
        self.assertIn("def checkout(cart):", out)
        self.assertIn("calls:", out)
        self.assertNotIn("sum(price_with_tax", out)

    def test_fit_chat(self):
        long = CODE * 6
        msgs = [{"role": "system", "content": "You help with a shop's code."},
                {"role": "user", "content": "Here is the code:\n" + long},
                {"role": "assistant", "content": "Got it."},
                {"role": "user", "content": "What does shipping cost for 3 kg?"},
                {"role": "assistant", "content": "7.0"},
                {"role": "user", "content": "And why is checkout wrong?"}]
        out = F.fit_chat(msgs, budget=900, keep_last=3)
        tot = sum(T.count(m["content"]) + 4 for m in out)
        self.assertLessEqual(tot, 900)
        self.assertEqual(out[0], msgs[0])
        self.assertEqual(out[-3:], msgs[-3:])

    def test_in_apsi(self):
        from apsi.interpreter import run_source
        env = run_source('use tokens\nlet f = tokens.focus("alpha beta gamma. " * 400, "beta", 50)\n'
                         'let ok = f.tokens <= 50\nlet back = tokens.unfold(f)\n')
        self.assertTrue(env.lookup("ok"))
        self.assertEqual(env.lookup("back"), "alpha beta gamma. " * 400)


if __name__ == "__main__":
    unittest.main()
