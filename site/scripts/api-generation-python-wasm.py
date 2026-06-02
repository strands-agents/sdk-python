# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pydoc-markdown>=4.8.2",
# ]
# ///
"""Generate markdown documentation for strands-agents-wasm SDK using pydoc-markdown.

This script generates per-module markdown files in the .build/api-docs/python-wasm/ directory.
Unlike the main Python SDK generation, no clone step is needed because strands-py-wasm/ lives
in the monorepo.

Usage:
    uv run scripts/api-generation-python-wasm.py   # if uv is available
    pip install pydoc-markdown && python scripts/api-generation-python-wasm.py  # fallback
"""

import shutil
from pathlib import Path

from pydoc_markdown import PydocMarkdown
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer
from pydoc_markdown.contrib.processors.filter import FilterProcessor
from pydoc_markdown.contrib.processors.crossref import CrossrefProcessor
from pydoc_markdown.contrib.processors.smart import SmartProcessor
from pydoc_markdown.contrib.source_linkers.git import GitSourceLinker
import docspec


class CustomGitSourceLinker(GitSourceLinker):
    """Custom source linker that returns 'Defined in: [path:line](url)' format."""

    def get_source_url(self, obj: docspec.ApiObject) -> str | None:
        url = super().get_source_url(obj)
        if not url or not obj.location:
            return None

        path = obj.location.filename
        if "src/" in path:
            path = "src/" + path.split("src/")[-1]

        lineno = obj.location.lineno
        return f"Defined in: [{path}:{lineno}]({url})"


def generate_docs():
    input_path = "../strands-py-wasm/src"
    output_path = "./.build/api-docs/python-wasm"

    output_dir = Path(output_path)

    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Deleted existing output directory: {output_dir}")

    output_dir.mkdir(exist_ok=True, parents=True)

    session = PydocMarkdown()

    loader = PythonLoader(
        search_path=[input_path],
        packages=["strands"],
    )
    session.loaders = [loader]

    session.processors = [
        FilterProcessor(skip_empty_modules=True),
        CrossrefProcessor(),
        SmartProcessor(),
    ]

    renderer = MarkdownRenderer(
        render_module_header=False,
        descriptive_class_title="",
        add_module_prefix=True,
        render_toc=False,
        source_linker=CustomGitSourceLinker(
            root="../strands-py-wasm/src",
            url_template="https://github.com/strands-agents/sdk-python/blob/main/strands-py-wasm/src/{path}#L{lineno}",
            use_branch=False,
        ),
        source_format="{url}",
    )
    session.renderer = renderer

    modules = session.load_modules()
    session.process(modules)

    excluded_modules: set[str] = set()

    module_files = []

    for module in modules:
        module_name = module.name

        # Skip private/internal modules
        parts = module_name.split(".")
        if any(part.startswith("_") for part in parts):
            print(f"Skipping private module: {module_name}")
            continue

        if module_name in excluded_modules:
            print(f"Skipping excluded module: {module_name}")
            continue

        # Skip any _generated submodules
        if "._generated" in module_name:
            print(f"Skipping generated module: {module_name}")
            continue

        filename = f"{module_name}.mdx"
        filepath = output_dir / filename
        slug = f"docs/api/python-wasm/{module_name}"

        content = renderer.render_to_string([module])

        content = f"""
---
title: {module_name}
slug:  {slug}
editUrl: false
---
{content}
""".strip()

        if content.strip():
            content = content.replace("{", "\\{").replace("<A2A", "&gt;A2A")
            filepath.write_text(content, encoding="utf-8")
            module_files.append((module_name, str(filepath.relative_to(output_dir))))
            print(f"Generated: {filepath}")

    print(f"\nTotal modules documented: {len(module_files)}")


if __name__ == "__main__":
    generate_docs()
