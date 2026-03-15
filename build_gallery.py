#!/usr/bin/env python3
"""
build_gallery.py
────────────────
Scans the photos/ folder, parses filenames, and regenerates the PHOTOS
array inside gallery.html.

Naming convention
─────────────────
  CategoryName-001.jpg    →  category "Category Name", title "Category Name 1"
  Singapore-003.png       →  category "Singapore",     title "Singapore 3"
  LabLife-012.webp        →  category "Lab Life",       title "Lab Life 12"

Rules:
  • The filename (without extension) must match:  <name>-<digits>
  • <name> is split on CamelCase humps and hyphens/underscores into title-case words.
  • The number after the last hyphen is used for sorting and appended to the title.
  • Any file that doesn't match the pattern is skipped (with a warning).
  • Supported extensions: jpg, jpeg, png, gif, webp, avif, svg

Usage:
  python3 build_gallery.py          # scans photos/, updates gallery.html
  python3 build_gallery.py --dry    # preview without writing

"""

import os
import re
import sys
import json

PHOTOS_DIR   = os.path.join(os.path.dirname(__file__), "photos")
GALLERY_HTML = os.path.join(os.path.dirname(__file__), "gallery.html")
EXTENSIONS   = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif", ".svg"}
START_MARKER = "// ── PHOTOS:START"
END_MARKER   = "// ── PHOTOS:END"


def camel_to_words(s: str) -> str:
    """LabLife → Lab Life,  NYCChase → NYC Chase"""
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    return s

def parse_filename(filename: str):
    """
    Returns (src, category, title, number) or None if filename doesn't match.
    """
    stem, ext = os.path.splitext(filename)
    if ext.lower() not in EXTENSIONS:
        return None

    m = re.match(r'^(.+)-(\d+)$', stem)
    if not m:
        return None

    raw_name = m.group(1)
    number   = int(m.group(2))

    # Normalise separators → spaces, then apply CamelCase split
    name_clean = raw_name.replace("_", " ").replace("-", " ")
    # Re-join so CamelCase within a token still works
    name_clean = " ".join(tok for tok in name_clean.split()) # camel_to_words(tok)
    # Title-case each word
    category = " ".join(w.capitalize() for w in name_clean.split())
    title    = f"{category} {number}"

    return {
        "src":      f"photos/{filename}",
        "title":    title,
        "desc":     "",
        "category": category,
        "fav":      False,
        "_sort_key": (category, number),
    }


def scan_photos():
    if not os.path.isdir(PHOTOS_DIR):
        print(f"⚠  photos/ folder not found at: {PHOTOS_DIR}")
        return []

    entries = []
    for fname in sorted(os.listdir(PHOTOS_DIR)):
        result = parse_filename(fname)
        if result is None:
            if not fname.startswith("."):
                print(f"   skip (no match): {fname}")
            continue
        entries.append(result)
        print(f"   OK  {fname:40s}  ->  [{result['category']}]  '{result['title']}'")

    entries.sort(key=lambda e: e["_sort_key"])
    for e in entries:
        del e["_sort_key"]
    return entries


def render_photos_js(photos):
    if not photos:
        return "const PHOTOS = [];"

    lines = ["const PHOTOS = ["]
    for p in photos:
        lines.append("  {")
        lines.append(f"    src:      {json.dumps(p['src'])},")
        lines.append(f"    title:    {json.dumps(p['title'])},")
        lines.append(f"    desc:     {json.dumps(p['desc'])},")
        lines.append(f"    category: {json.dumps(p['category'])},")
        lines.append(f"    fav:      {'true' if p['fav'] else 'false'},")
        lines.append("  },")
    lines.append("];")
    return "\n".join(lines)


def update_gallery_html(photos_js: str, dry: bool):
    with open(GALLERY_HTML, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the markers
    start_idx = content.find(START_MARKER)
    end_idx   = content.find(END_MARKER)
    if start_idx == -1 or end_idx == -1:
        print("✗  Could not find PHOTOS:START / PHOTOS:END markers in gallery.html")
        sys.exit(1)

    # Find end of the START marker line
    start_line_end = content.index("\n", start_idx) + 1
    # Build replacement
    new_block = (
        content[:start_line_end]
        + photos_js + "\n"
        + content[end_idx:]
    )

    if dry:
        print(photos_js)
    else:
        with open(GALLERY_HTML, "w", encoding="utf-8") as f:
            f.write(new_block)
        print(f"\nOK  gallery.html updated ({len(photos)} photo(s))")


if __name__ == "__main__":
    dry = "--dry" in sys.argv

    print(f"Scanning {PHOTOS_DIR} …\n")
    photos = scan_photos()
    print(f"\nFound {len(photos)} matching photo(s).")

    photos_js = render_photos_js(photos)
    update_gallery_html(photos_js, dry)
