import os
import shutil

# Copy CHANGELOG file to 'docs' and change its extension to '.qmd'
shutil.copyfile("../CHANGELOG.md", "CHANGELOG.md")
os.rename("CHANGELOG.md", "changelog.qmd")

# Remove TOC from it
lines = """---
toc: false
pagetitle: "Changelog"
---
"""
with open("changelog.qmd", "r+") as file:
    file_data = file.read()
    file.seek(0, 0)
    file.write(lines + "\n" + file_data)
