"""Create the blog as a Notion page from notion_blocks.json.

Usage:  NOTION_TOKEN=ntn_... python3 notion_publish.py --parent <page id or URL> [--blocks notion_blocks.json]
The integration behind the token must have access to the parent page (page menu -> Connections). Images and the
cover are uploaded with Notion's file-upload API, so nothing has to be hosted elsewhere.
"""
import argparse, json, os, re, subprocess, sys

API = 'https://api.notion.com/v1'
HERE = os.path.dirname(os.path.abspath(__file__))

def call(token, method, path, body=None, extra=None):
    cmd = ['curl', '-s', '-m', '60', '-X', method, API + path, '-H', 'Authorization: Bearer ' + token, '-H', 'Notion-Version: 2022-06-28']
    if body is not None: cmd += ['-H', 'Content-Type: application/json', '-d', json.dumps(body)]
    if extra: cmd += extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    try: d = json.loads(r.stdout)
    except Exception: sys.exit('bad reply from ' + path + ': ' + r.stdout[:300])
    if d.get('object') == 'error': sys.exit('Notion error on ' + path + ': ' + d.get('message', ''))
    return d

def upload(token, path):
    name = os.path.basename(path); ext = name.rsplit('.', 1)[-1].lower()
    ctype = {'png': 'image/png', 'gif': 'image/gif', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'mp4': 'video/mp4'}[ext]
    fu = call(token, 'POST', '/file_uploads', {'filename': name, 'content_type': ctype})
    r = subprocess.run(['curl', '-s', '-m', '300', '-X', 'POST', fu['upload_url'], '-H', 'Authorization: Bearer ' + token, '-H', 'Notion-Version: 2022-06-28', '-F', 'file=@' + path + ';type=' + ctype], capture_output=True, text=True)
    d = json.loads(r.stdout)
    if d.get('status') != 'uploaded': sys.exit('upload failed for ' + name + ': ' + r.stdout[:300])
    return fu['id']

def page_id(s):
    m = re.search(r'([0-9a-f]{32})', s.replace('-', ''))
    if not m: sys.exit('cannot find a page id in ' + s)
    h = m.group(1); return f'{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}'

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--parent', required=True); ap.add_argument('--blocks', default=os.path.join(HERE, 'notion_blocks.json'))
    a = ap.parse_args(); token = os.environ.get('NOTION_TOKEN') or sys.exit('set NOTION_TOKEN')
    spec = json.load(open(a.blocks)); children = spec['children']
    for b in children:   # upload local images
        if b['type'] == 'image' and b['image']['external']['url'].startswith('LOCAL:'):
            local = b['image']['external']['url'][6:]
            fid = upload(token, os.path.join(HERE, local)); print('uploaded', local)
            b['image'] = {'type': 'file_upload', 'file_upload': {'id': fid}, 'caption': b['image']['caption']}
    page = {'parent': {'page_id': page_id(a.parent)}, 'icon': {'type': 'emoji', 'emoji': spec['icon']},
            'properties': {'title': {'title': [{'type': 'text', 'text': {'content': spec['title']}}]}}, 'children': children[:100]}
    if spec.get('cover'):
        cid = upload(token, os.path.join(HERE, spec['cover'])); page['cover'] = {'type': 'file_upload', 'file_upload': {'id': cid}}
    d = call(token, 'POST', '/pages', page); pid = d['id']; print('created page', d.get('url'))
    for i in range(100, len(children), 100):
        call(token, 'PATCH', '/blocks/' + pid + '/children', {'children': children[i:i + 100]}); print('appended blocks', i, '-', min(i + 100, len(children)))
    print('done:', d.get('url'))

if __name__ == '__main__': main()
