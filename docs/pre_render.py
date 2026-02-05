import os
import shutil


def starts_with_anchor(line: str) -> bool:
    return line.lstrip().startswith('<a id="') and line.rstrip().endswith('"></a>')


def starts_with_heading(line: str) -> bool:
    return line.lstrip().startswith("# ")


def extract_id(anchor_line: str) -> str:
    return anchor_line.strip().split('id="', 1)[1].split('"', 1)[0]


def convert_anchor_headings(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines) and starts_with_anchor(line) and starts_with_heading(lines[i + 1]):
            anchor_id = extract_id(line)
            heading = lines[i + 1].rstrip("\n")
            out.append(f"{heading} {{{'#' + anchor_id}}}\n")
            i += 2
            continue
        out.append(line)
        i += 1
    return "".join(out)


# Copy CHANGELOG file to 'docs' and change its extension to '.qmd'
shutil.copyfile("../CHANGELOG.md", "CHANGELOG.md")
os.replace("CHANGELOG.md", "changelog.qmd")

# Remove TOC from it
lines = """---
toc: false
pagetitle: "Changelog"
---
"""
with open("changelog.qmd", "r+") as changelog_file:
    file_data = convert_anchor_headings(changelog_file.read())
    changelog_file.seek(0, 0)
    changelog_file.write(lines + "\n" + file_data)
    changelog_file.truncate()
