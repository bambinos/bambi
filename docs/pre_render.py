import os
import re
import shutil
from pathlib import Path

import yaml

build_dir = Path(os.environ["QUARTO_PROJECT_DIR"])
docs_dir = build_dir.parents[1]

# Great Docs discovers Markdown pages, so copy the Jupyter notebooks separately.
for source in (docs_dir / "notebooks").iterdir():
    if source.suffix in {".ipynb", ".ejs", ".yml"}:
        shutil.copy2(source, build_dir / "notebooks" / source.name)

# Copy the report page and replace the software citation with the paper citation.
shutil.copy2(docs_dir / "coc-violation.qmd", build_dir / "coc-violation.qmd")
shutil.copy2(docs_dir / "citation.qmd", build_dir / "citation.qmd")

# Keep Quarto's sidebar collapse controls enabled.
sidebar_script = build_dir / "sidebar-wrap.js"
sidebar_script.write_text(sidebar_script.read_text().replace("disableSidebarCollapse();", ""))

# Remove AI / Agents and place Community before Developers on the homepage.
homepage = build_dir / "index.qmd"
content = re.sub(r"\n#### AI / Agents\n.*?(?=\n#### |\n:::)", "", homepage.read_text(), flags=re.S)
content = re.sub(
    r"(\n#### Developers\n.*?)(\n#### Community\n.*?)(?=\n#### |\n:::)",
    r"\2\1",
    content,
    flags=re.S,
)
homepage.write_text(content)

# Load the generated configuration, which Quarto reloads after pre-render.
quarto_path = build_dir / "_quarto.yml"
header = quarto_path.read_text().split("---", 1)[0]
with quarto_path.open() as stream:
    config = yaml.safe_load(stream)
website = config["website"]

# Use local favicon URLs during preview.
for include in config["format"]["html"].get("include-in-header", []):
    if "favicon" in include.get("text", "") or "apple-touch-icon" in include.get("text", ""):
        include["text"] = include["text"].replace(website["site-url"], "/")

# Point Quickstart to the introductory notebook.
website["navbar"]["left"] = [
    {"text": "Quickstart", "href": "notebooks/getting_started.ipynb"},
    *[item for item in website["navbar"]["left"] if item.get("text") != "Quickstart"],
]

# Credit the developers and Great Docs in the footer.
website["page-footer"]["center"] = (
    "Developed by <strong>Bambi developers</strong>.<br>"
    "Site created with <strong>Great Docs</strong>."
)

# Include the sidebar layout script once, even on repeated renders.
includes = config["format"]["html"].setdefault("include-after-body", [])
includes[:] = [item for item in includes if 'id="bambi-sidebars"' not in item.get("text", "")]
includes.append(
    {
        "text": '<script id="bambi-sidebars">\n'
        + (docs_dir / "sidebars.js").read_text()
        + "\n</script>"
    }
)

# Match the examples sidebar to the gallery categories and order.
with (docs_dir / "notebooks" / "gallery.yml").open() as stream:
    gallery = yaml.safe_load(stream)
for sidebar in website["sidebar"]:
    if sidebar["id"] == "notebooks":
        sidebar["collapse-level"] = 1
        sidebar["contents"] = [
            {"text": "Examples gallery", "href": "notebooks/index.qmd"},
            *[
                {
                    "section": group["category"],
                    "contents": ["notebooks/" + tile["href"] for tile in group["tiles"]],
                }
                for group in gallery
            ],
        ]

# Save the configuration while preserving the generated-file notice.
quarto_path.write_text(
    header + "---\n" + yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
)
