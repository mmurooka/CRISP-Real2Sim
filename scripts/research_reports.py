"""Write numbered, standalone local research reports and a newest-first index."""
from datetime import datetime
from html import escape
import json
from pathlib import Path
import re

STYLE = """
body{font:17px/1.8 system-ui,sans-serif;color:#1d2c39;background:#f5f7fa;margin:0}
main{max-width:1080px;margin:auto;padding:36px 28px;background:white;min-height:100vh}
h1{font-size:30px;line-height:1.4}h2{font-size:23px;margin-top:40px;border-bottom:2px solid #dbe5ee}
a{color:#145f9a}table{border-collapse:collapse;width:100%;font-size:15px}
th,td{border:1px solid #d5dde5;padding:12px;text-align:left;vertical-align:top;overflow-wrap:anywhere}
th{background:#eef3f8}code{font-size:14px;overflow-wrap:anywhere}pre{white-space:pre-wrap;background:#eef3f8;padding:16px}
video,img{max-width:100%;border-radius:8px}video{background:#111;width:100%}
.status{padding:16px;border-left:5px solid #b77913;background:#fff8e7}.ok{border-color:#14805e;background:#eaf6f1}
.meta{color:#607084;font-size:14px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:24px}
li{margin:8px 0}figure{margin:20px 0}figcaption{font-size:14px;color:#607084}
"""


def page(title, content):
    return f'<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{STYLE}</style><main>{content}</main></html>'


def publish_report(root, slug, title, body):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r"[a-z0-9_-]+", slug):
        raise ValueError("report slug must be a safe filename component")
    manifest = root / "reports.json"
    entries = json.loads(manifest.read_text()) if manifest.exists() else []
    existing = [int(p.name.split('_')[0]) for p in root.glob('[0-9]*_*.html')]
    number = max(existing, default=0) + 1
    filename = f"{number:03d}_{slug}.html"
    created = datetime.now().astimezone().isoformat(timespec="seconds")
    content = f'<p><a href="index.html">報告一覧へ</a></p><h1>{number:03d} — {escape(title)}</h1><p class="meta">{escape(created)}</p>{body}'
    path = root / filename
    with path.open("x") as handle:
        handle.write(page(title, content))
    entries.append(dict(number=number, filename=filename, title=title, created=created))
    manifest.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + '\n')
    links = ''.join(f'<li><a href="{e["filename"]}">{e["number"]:03d} — {escape(e["title"])}</a><div class="meta">{escape(e["created"])}</div></li>'
                    for e in sorted(entries, key=lambda e:e['number'], reverse=True))
    (root / "index.html").write_text(page("全身接触HOI研究・報告一覧",
        '<h1>全身接触HOI研究・報告一覧</h1><p>動画からの復元 → 全身接触を保つG1リターゲティング → RL → 実機評価</p>'
        '<p>新しい報告から順に表示しています。実動画の結果と合成テストを区別して記録します。</p><ol reversed>' + links + '</ol>'))
    return path
