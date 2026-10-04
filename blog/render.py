"""Render content.py into loopback_blog.md (Notion-flavoured Markdown), loopback_blog.html (preview) and
notion_blocks.json (Notion API blocks; images carry local paths that notion_publish.py uploads)."""
import html, json, os, re
import content as C

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_RE = re.compile(r'(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`|\$[^$]+?\$|\[[^\]]+?\]\([^)]+?\))')

def tokens(s):
    """[(kind, text, extra)] with kind in text/bold/italic/code/math/link; plain text may contain any character."""
    out, pos = [], 0
    for m in TOKEN_RE.finditer(s):
        if m.start() > pos: out.append(('text', s[pos:m.start()], None))
        part = m.group(0)
        if part.startswith('**'): out.append(('bold', part[2:-2], None))
        elif part.startswith('`'): out.append(('code', part[1:-1], None))
        elif part.startswith('$'): out.append(('math', part[1:-1], None))
        elif part.startswith('['):
            lm = re.match(r'\[(.+?)\]\((.+?)\)', part); out.append(('link', lm.group(1), lm.group(2)))
        else: out.append(('italic', part[1:-1], None))
        pos = m.end()
    if pos < len(s): out.append(('text', s[pos:], None))
    return out

# ---------------- Markdown ----------------
def md_rich(s): return s
def md_block(b, out):
    k = b[0]
    if k == 'p': out.append(b[1] + '\n')
    elif k in ('h1', 'h2', 'h3'): out.append('#' * int(k[1]) + ' ' + b[1] + '\n')
    elif k == 'ul': out.extend('- ' + x for x in b[1]); out.append('')
    elif k == 'callout':
        lines = []
        for sub in b[2]:
            if sub[0] == 'p': lines.append(sub[1]); lines.append('')
            elif sub[0] == 'ul': lines.extend('- ' + x for x in sub[1]); lines.append('')
        out.append('> ' + b[1] + ' ' + '\n> '.join(lines).rstrip('\n> ') + '\n')
    elif k == 'eq': out.append('$$\n' + b[1] + '\n$$\n')
    elif k == 'img': out.append('![' + b[2].split('.')[0] + '](' + b[1] + ')\n\n*' + b[2] + '*\n')
    elif k == 'table':
        hdr, rows = b[1], b[2]
        out.append('| ' + ' | '.join(hdr) + ' |'); out.append('|' + '---|' * len(hdr))
        out.extend('| ' + ' | '.join(r) + ' |' for r in rows); out.append('')
    elif k == 'code': out.append('```' + b[1] + '\n' + b[2] + '\n```\n')
    elif k == 'divider': out.append('---\n')
    elif k == 'bookmark': out.append(b[1] + '\n')
    elif k == 'toc': pass
    elif k == 'qa': out.append('**' + b[1] + '**\n\n' + b[2] + '\n')

def render_md():
    out = ['# ' + C.TITLE + '\n']
    for b in C.BLOCKS: md_block(b, out)
    return '\n'.join(out)

# ---------------- HTML ----------------
def slug(t):
    t = re.sub(r'[^A-Za-z0-9 ]+', '', re.sub(r'\$[^$]*\$', '', t)).strip().lower(); return re.sub(r'\s+', '-', t)[:60] or 'section'
def h_rich(s):
    o = ''
    for kind, t, extra in tokens(s):
        e = html.escape(t)
        if kind == 'bold': o += '<strong>' + e + '</strong>'
        elif kind == 'italic': o += '<em>' + e + '</em>'
        elif kind == 'code': o += '<code>' + e + '</code>'
        elif kind == 'math': o += '\\(' + e + '\\)'
        elif kind == 'link': o += '<a href="' + html.escape(extra) + '">' + e + '</a>'
        else: o += e
    return o
def h_block(b, out):
    k = b[0]
    if k == 'p': out.append('<p>' + h_rich(b[1]) + '</p>')
    elif k in ('h1', 'h2', 'h3'): n = {'h1': 2, 'h2': 3, 'h3': 4}[k]; out.append(f'<h{n} id="' + slug(b[1]) + '">' + h_rich(b[1]) + f'</h{n}>')
    elif k == 'ul': out.append('<ul>' + ''.join('<li>' + h_rich(x) + '</li>' for x in b[1]) + '</ul>')
    elif k == 'callout':
        inner = []
        for sub in b[2]: h_block(sub, inner)
        out.append('<div class="callout"><div class="callout-icon">' + b[1] + '</div><div class="callout-body">' + ''.join(inner) + '</div></div>')
    elif k == 'eq': out.append('<div class="eq">\\[' + html.escape(b[1]) + '\\]</div>')
    elif k == 'img':
        tag = '<img src="' + b[1] + '" alt="' + html.escape(b[2][:80]) + '">'
        out.append('<figure>' + tag + '<figcaption>' + h_rich(b[2]) + '</figcaption></figure>')
    elif k == 'table':
        out.append('<div class="tablewrap"><table><thead><tr>' + ''.join('<th>' + h_rich(c) + '</th>' for c in b[1]) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + h_rich(c) + '</td>' for c in r) + '</tr>' for r in b[2]) + '</tbody></table></div>')
    elif k == 'code': out.append('<pre><code>' + html.escape(b[2]) + '</code></pre>')
    elif k == 'divider': out.append('<hr>')
    elif k == 'bookmark': out.append('<p class="bookmark"><a href="' + html.escape(b[1]) + '">' + html.escape(b[1]) + '</a></p>')
    elif k == 'toc': out.append('<nav class="toc"><div class="toc-title">Contents</div><ol>' + ''.join('<li><a href="#' + slug(x[1]) + '">' + h_rich(x[1]) + '</a></li>' for x in C.BLOCKS if x[0] == 'h1') + '</ol></nav>')
    elif k == 'qa': out.append('<details class="qa"><summary>' + h_rich(b[1]) + '</summary><p>' + h_rich(b[2]) + '</p></details>')

CSS = """
:root { --ink:#1b1f24; --muted:#5b6470; --line:#e6e8eb; --navy:#0F4D92; --rose:#B64342; --bg:#ffffff; --callout:#f4f6f9; --accent:#0645ad; }
@media (prefers-color-scheme: dark) { :root:not([data-theme=light]) { --ink:#e8eaed; --muted:#a5adb8; --line:#2d3239; --navy:#6ea8ff; --bg:#111417; --callout:#1a1f26; --accent:#8ab4ff; } }
:root[data-theme=dark] { --ink:#e8eaed; --muted:#a5adb8; --line:#2d3239; --navy:#6ea8ff; --bg:#111417; --callout:#1a1f26; --accent:#8ab4ff; }
* { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height:1.65; font-size:17px; }
.cover { width:100%; height:clamp(160px, 30vw, 340px); object-fit:cover; display:block; }
.page { max-width:820px; margin:0 auto; padding:0 16px 80px; }
.icon { font-size:64px; line-height:1; margin:-32px 0 8px; }
h1.title { font-family:"Times New Roman", Times, serif; font-size:clamp(28px, 4.4vw, 42px); line-height:1.15; margin:8px 0 18px; letter-spacing:-0.01em; }
h2 { font-family:"Times New Roman", Times, serif; font-size:30px; margin:48px 0 12px; line-height:1.2; } h3 { font-size:21px; margin:30px 0 8px; } h4 { font-size:18px; margin:26px 0 6px; }
p { margin:0 0 14px } a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline }
ul { padding-left:1.3em; margin:0 0 14px } li { margin:4px 0 }
.callout { display:flex; gap:14px; background:var(--callout); border:1px solid var(--line); border-radius:10px; padding:16px 18px; margin:14px 0 18px; } .callout-icon { font-size:24px; line-height:1.3 } .callout-body { min-width:0; flex:1 } .callout-body p:last-child, .callout-body ul:last-child { margin-bottom:0 }
figure { margin:22px 0 } figure img, figure video { width:100%; height:auto; border-radius:8px; border:1px solid var(--line); display:block; background:#fff } figcaption { color:var(--muted); font-size:14.5px; margin-top:8px; line-height:1.5 }
.eq { overflow-x:auto; padding:6px 0; margin:6px 0 14px }
.tablewrap { overflow-x:auto; margin:10px 0 18px } table { border-collapse:collapse; width:100%; font-size:15px; font-variant-numeric:tabular-nums } th, td { padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top } th { color:var(--muted); font-weight:600; font-size:13.5px; text-transform:uppercase; letter-spacing:.03em }
pre { background:var(--callout); border:1px solid var(--line); border-radius:8px; padding:14px 16px; overflow-x:auto; font-size:13.5px; line-height:1.5 } code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:0.92em }
hr { border:0; border-top:1px solid var(--line); margin:36px 0 }
.bookmark a { display:block; border:1px solid var(--line); border-radius:8px; padding:12px 14px; background:var(--callout) }
.meta p { margin:0 0 6px; color:var(--muted); font-size:15.5px } .meta strong { color:var(--ink) }
.toc { background:var(--callout); border:1px solid var(--line); border-radius:10px; padding:14px 18px; margin:18px 0 } .toc-title { font-weight:600; font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin-bottom:6px } .toc ol { margin:0; padding-left:1.2em; columns:2; column-gap:28px } .toc li { margin:3px 0; break-inside:avoid }
details.qa { border:1px solid var(--line); border-radius:8px; padding:10px 14px; margin:8px 0; background:var(--bg) } details.qa summary { cursor:pointer; font-weight:600 } details.qa p { margin:10px 0 2px }
@media (max-width:640px) { .toc ol { columns:1 } }
"""
def render_html():
    body = []
    for b in C.BLOCKS: h_block(b, body)
    cover = '<img class="cover" src="' + C.COVER + '" alt="">' if C.COVER else ''
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<title>Loopback Transformer blog</title>\n<meta name="description" content="Blog post introducing the Loopback Transformer: a looped Transformer that stores block updates, keeps the input on its own path, and lets decay and attention set and allocate the update weight.">\n'
            '<style>' + CSS + '</style>\n'
            '<script>window.MathJax={tex:{inlineMath:[["\\\\(","\\\\)"]],displayMath:[["\\\\[","\\\\]"]]},svg:{fontCache:"global"}};</script>\n'
            '<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.js" async></script>\n</head>\n<body>\n'
            + cover + '<div class="page"><div class="icon">' + C.ICON + '</div><h1 class="title">' + html.escape(C.TITLE) + '</h1><div class="meta">' + ''.join(body[:3]) + '</div>' + ''.join(body[3:]) + '</div>\n</body>\n</html>\n')
    return page

# ---------------- Notion API blocks ----------------
def n_rich(s):
    out = []
    for kind, t, extra in tokens(s):
        if kind == 'math': out.append({'type': 'equation', 'equation': {'expression': t}}); continue
        item = {'type': 'text', 'text': {'content': t}}
        if kind == 'link': item['text']['link'] = {'url': extra}
        ann = {}
        if kind == 'bold': ann['bold'] = True
        if kind == 'italic': ann['italic'] = True
        if kind == 'code': ann['code'] = True
        if ann: item['annotations'] = ann
        out.append(item)
    return out
def n_block(b):
    k = b[0]
    if k == 'p': return {'object': 'block', 'type': 'paragraph', 'paragraph': {'rich_text': n_rich(b[1])}}
    if k in ('h1', 'h2', 'h3'): t = 'heading_' + k[1]; return {'object': 'block', 'type': t, t: {'rich_text': n_rich(b[1])}}
    if k == 'ul': return [{'object': 'block', 'type': 'bulleted_list_item', 'bulleted_list_item': {'rich_text': n_rich(x)}} for x in b[1]]
    if k == 'callout':
        subs = []
        for sub in b[2]:
            r = n_block(sub); subs.extend(r if isinstance(r, list) else [r])
        first, rest = subs[0], subs[1:]
        rt = first[first['type']]['rich_text']
        return {'object': 'block', 'type': 'callout', 'callout': {'rich_text': rt, 'icon': {'type': 'emoji', 'emoji': b[1]}, 'color': 'gray_background', 'children': rest}} if first['type'] == 'paragraph' else \
               {'object': 'block', 'type': 'callout', 'callout': {'rich_text': [], 'icon': {'type': 'emoji', 'emoji': b[1]}, 'color': 'gray_background', 'children': subs}}
    if k == 'eq': return {'object': 'block', 'type': 'equation', 'equation': {'expression': b[1]}}
    if k == 'img': return {'object': 'block', 'type': 'image', 'image': {'type': 'external', 'external': {'url': 'LOCAL:' + b[1]}, 'caption': n_rich(b[2])}}
    if k == 'table':
        rows = [b[1]] + b[2]
        return {'object': 'block', 'type': 'table', 'table': {'table_width': len(b[1]), 'has_column_header': True, 'has_row_header': False,
                'children': [{'object': 'block', 'type': 'table_row', 'table_row': {'cells': [n_rich(c) for c in r]}} for r in rows]}}
    if k == 'code': return {'object': 'block', 'type': 'code', 'code': {'rich_text': [{'type': 'text', 'text': {'content': b[2]}}], 'language': b[1]}}
    if k == 'divider': return {'object': 'block', 'type': 'divider', 'divider': {}}
    if k == 'bookmark': return {'object': 'block', 'type': 'bookmark', 'bookmark': {'url': b[1]}}
    if k == 'toc': return {'object': 'block', 'type': 'table_of_contents', 'table_of_contents': {'color': 'gray'}}
    if k == 'qa': return {'object': 'block', 'type': 'toggle', 'toggle': {'rich_text': n_rich(b[1]), 'children': [{'object': 'block', 'type': 'paragraph', 'paragraph': {'rich_text': n_rich(b[2])}}]}}
    raise ValueError(k)
def render_notion():
    blocks = []
    for b in C.BLOCKS:
        r = n_block(b); blocks.extend(r if isinstance(r, list) else [r])
    return {'title': C.TITLE, 'icon': C.ICON, 'cover': C.COVER, 'children': blocks}

if __name__ == '__main__':
    open(os.path.join(HERE, 'loopback_blog.md'), 'w').write(render_md())
    open(os.path.join(HERE, 'loopback_blog.html'), 'w').write(render_html())
    json.dump(render_notion(), open(os.path.join(HERE, 'notion_blocks.json'), 'w'), ensure_ascii=False, indent=1)
    print('wrote loopback_blog.md, loopback_blog.html, notion_blocks.json')
