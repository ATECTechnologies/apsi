"""Generate the editor integrations that are derived from the VS Code extension.

    python tools/build_editors.py

The TextMate grammar (editors/vscode/syntaxes/apsi.tmLanguage.json) and the snippets
(editors/vscode/snippets/apsi.json) are the single source of truth. This script writes:

  editors/sublime/Apsi/Apsi.tmLanguage                 Sublime Text syntax (plist)
  editors/sublime/Apsi/snippets/*.sublime-snippet      Sublime Text snippets
  editors/textmate/Apsi.tmbundle/Syntaxes/Apsi.tmLanguage   TextMate / JetBrains bundle
  editors/textmate/Apsi.tmbundle/Snippets/*.tmSnippet
"""
import json
import os
import plistlib
import re
import shutil
import uuid
from xml.sax.saxutils import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ED = os.path.join(ROOT, "editors")


def load(rel):
    with open(os.path.join(ED, rel), encoding="utf-8") as fh:
        return json.load(fh)


def clean_dir(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)


def slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()


def main():
    grammar = load("vscode/syntaxes/apsi.tmLanguage.json")
    grammar.pop("$schema", None)
    grammar["uuid"] = "8F6B6D7C-6A0B-4F2E-9B7E-4F2C1A0B1E57"
    grammar.setdefault("fileTypes", ["apsi"])
    for target in ("sublime/Apsi/Apsi.tmLanguage", "textmate/Apsi.tmbundle/Syntaxes/Apsi.tmLanguage"):
        path = os.path.join(ED, *target.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            plistlib.dump(grammar, fh, sort_keys=False)

    snippets = load("vscode/snippets/apsi.json")
    sub_dir = os.path.join(ED, "sublime", "Apsi", "snippets")
    tm_dir = os.path.join(ED, "textmate", "Apsi.tmbundle", "Snippets")
    clean_dir(sub_dir)
    clean_dir(tm_dir)
    for name, s in snippets.items():
        body = "\n".join(s["body"]) if isinstance(s["body"], list) else s["body"]
        # VS Code choices ${1|a,b|} are not supported elsewhere: keep the first choice as a placeholder
        body = re.sub(r"\$\{(\d+)\|([^,|}]*)[^}]*\|\}", r"${\1:\2}", body)
        prefix = s["prefix"]
        desc = s.get("description", name)
        with open(os.path.join(sub_dir, slug(name) + ".sublime-snippet"), "w", encoding="utf-8") as fh:
            fh.write(f"<snippet>\n  <content><![CDATA[\n{body}\n]]></content>\n"
                     f"  <tabTrigger>{escape(prefix)}</tabTrigger>\n  <scope>source.apsi</scope>\n"
                     f"  <description>{escape(desc)}</description>\n</snippet>\n")
        with open(os.path.join(tm_dir, slug(name) + ".tmSnippet"), "wb") as fh:
            plistlib.dump({"name": desc, "tabTrigger": prefix, "scope": "source.apsi", "content": body,
                           "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, "apsi-snippet-" + name)).upper()}, fh)
    print(f"wrote 2 grammars and {len(snippets)} snippets × 2 into {ED}")


if __name__ == "__main__":
    main()
