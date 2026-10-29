"""Put the built page on the gh-pages branch, which GitHub Pages serves: index.html, the images and videos it references,
and .nojekyll. Run build.py first; afterwards push with `git push origin gh-pages`.

The commit is written through a temporary index, so the working tree, the checked-out branch and its index are untouched.
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = 'https://yancyou.github.io/Loopback_page/'


def git(*args, env=None, stdin=None):
    return subprocess.run(['git', '-C', HERE, *args], env=env, input=stdin, capture_output=True, text=True,
                          check=True).stdout.strip()


page = open(os.path.join(HERE, 'index.html'), encoding='utf-8').read()
# relative references, plus the absolute og:/twitter: image URLs on the Pages address
refs = set(re.findall(r'(?:src|href|poster|content)="(?:' + re.escape(SITE) + r')?((?:images|videos)/[^"#?]+)"', page))
files = ['index.html'] + sorted(refs)
missing = [f for f in files if not os.path.isfile(os.path.join(HERE, f))]
if missing:
    sys.exit('missing: ' + ', '.join(missing))

with tempfile.TemporaryDirectory() as tmp:
    env = dict(os.environ, GIT_INDEX_FILE=os.path.join(tmp, 'index'))
    for f in files:
        git('update-index', '--add', '--cacheinfo', '100644,%s,%s' % (git('hash-object', '-w', f), f), env=env)
    git('update-index', '--add', '--cacheinfo', '100644,%s,.nojekyll' % git('hash-object', '-w', '--stdin', stdin=''),
        env=env)
    tree = git('write-tree', env=env)

parent = subprocess.run(['git', '-C', HERE, 'rev-parse', '-q', '--verify', 'refs/heads/gh-pages'], capture_output=True,
                        text=True).stdout.strip()
if parent and git('rev-parse', parent + '^{tree}') == tree:
    sys.exit('gh-pages already holds this build')
commit = git('commit-tree', tree, *(['-p', parent] if parent else []),
             '-m', 'Publish the page built from ' + git('rev-parse', '--short', 'HEAD'))
git('update-ref', 'refs/heads/gh-pages', commit)
print('gh-pages -> %s (%d files); push with: git push origin gh-pages' % (commit[:7], len(files) + 1))
