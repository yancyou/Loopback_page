"""patch_hero.py -- replace the <figure class="prior-card"> block of page.html with the markup printed by hero_prior.py."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
page = os.path.join(HERE, "page.html")
src = open(page, encoding="utf-8").read()
a = src.index('<figure class="prior-card')
b = src.index("</figure>", a) + len("</figure>")
new = subprocess.run([sys.executable, os.path.join(HERE, "tools", "hero_prior.py")], check=True, capture_output=True, text=True).stdout.strip()
indent = src[src.rindex("\n", 0, a) + 1:a]
new = new.replace("\n", "\n" + indent)
open(page, "w", encoding="utf-8").write(src[:a] + new + src[b:])
print(f"replaced {b - a} chars with {len(new)}")
