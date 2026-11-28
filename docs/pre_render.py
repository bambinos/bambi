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
shutil.copy2(docs_dir / "contact.qmd", build_dir / "contact.qmd")

# Keep Quarto's sidebar collapse controls enabled.
sidebar_script = build_dir / "sidebar-wrap.js"
sidebar_script.write_text(sidebar_script.read_text().replace("disableSidebarCollapse();", ""))

# Great Docs has no setting to omit the AI / Agents sidebar section.
homepage = build_dir / "index.qmd"
content = re.sub(r"\n#### AI / Agents\n.*?(?=\n#### |\n:::)", "", homepage.read_text(), flags=re.S)
content = re.sub(
    r"(\n#### Developers\n.*?)(\n#### Community\n.*?)(?=\n#### |\n:::)",
    r"\2\1",
    content,
    flags=re.S,
)
homepage.write_text(content)

# Quarto reloads its configuration after pre-render scripts run.
quarto_path = build_dir / "_quarto.yml"
header = quarto_path.read_text().split("---", 1)[0]
with quarto_path.open() as stream:
    config = yaml.safe_load(stream)
website = config["website"]
for include in config["format"]["html"].get("include-in-header", []):
    if "favicon" in include.get("text", "") or "apple-touch-icon" in include.get("text", ""):
        include["text"] = include["text"].replace(website["site-url"], "/")
website["navbar"]["left"] = [
    {"text": "Quickstart", "href": "notebooks/getting_started.ipynb"},
    *[item for item in website["navbar"]["left"] if item.get("text") != "Quickstart"],
]
website["page-footer"]["center"] = (
    "Developed by <strong>Bambi developers</strong>.<br>"
    "Site created with <strong>Great Docs</strong>."
)
includes = config["format"]["html"].setdefault("include-after-body", [])
includes[:] = [item for item in includes if 'id="bambi-sidebars"' not in item.get("text", "")]
includes.append(
    {
        "text": '<script id="bambi-sidebars">\n'
        + (docs_dir / "sidebars.js").read_text()
        + "\n</script>"
    }
)

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
quarto_path.write_text(
    header + "---\n" + yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
)
