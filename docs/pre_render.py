import shutil

# Copy CHANGELOG file to 'docs'
shutil.copyfile("../CHANGELOG.md", "CHANGELOG.md")

# Remove TOC from it
lines = """---
toc: false
---
"""
with open("CHANGELOG.md", "r+") as file: 
    file_data = file.read() 
    file.seek(0, 0) 
    file.write(lines + "\n" + file_data) 