# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pydoc-markdown>=4.8.2",
# ]
# ///
"""Generate markdown documentation for strands-agents SDK using pydoc-markdown.

This script generates per-module markdown files in the .build/api-docs/python/ directory.

Usage:
    uv run scripts/api-generation-python.py   # if uv is available
    pip install pydoc-markdown && python scripts/api-generation-python.py  # fallback
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
        # Get the base URL from parent
        url = super().get_source_url(obj)
        if not url or not obj.location:
            return None

        # Extract path relative to src/
        path = obj.location.filename
        if "src/" in path:
            path = "src/" + path.split("src/")[-1]

        lineno = obj.location.lineno
        return f"Defined in: [{path}:{lineno}]({url})"


def generate_docs():
    input_path =  "./.build/sdk-python/strands-py/src"
    output_path = "./.build/api-docs/python"

    """Generate markdown documentation for all strands modules."""
    output_dir = Path(output_path)

    # Delete existing output directory to ensure clean generation
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Deleted existing output directory: {output_dir}")

    output_dir.mkdir(exist_ok=True, parents=True)

    # Configure the session
    session = PydocMarkdown()

    # Configure the Python loader
    loader = PythonLoader(
        search_path=[input_path],
        packages=["strands"],
    )
    session.loaders = [loader]

    # Configure processors (filter, crossref, smart)
    session.processors = [
        FilterProcessor(skip_empty_modules=True),
        CrossrefProcessor(),
        SmartProcessor(),
    ]

    # Configure the renderer
    renderer = MarkdownRenderer(
        render_module_header=False,
        descriptive_class_title="",
        add_module_prefix=True,
        render_toc=False,
        source_linker=CustomGitSourceLinker(
            root=".build/sdk-python/strands-py/src",
            url_template="https://github.com/strands-agents/sdk-python/blob/main/strands-py/src/{path}#L{lineno}",
            use_branch=False,
        ),
        source_format="{url}",  # URL already contains the full formatted string
    )
    session.renderer = renderer

    # Load and process modules
    modules = session.load_modules()
    session.process(modules)

    # Modules to exclude from documentation
    excluded_modules = {
        "strands.agent",  # Not useful, just re-exports
    }

    # Generate index file
    module_files = []

    # Write each module to a separate file
    for module in modules:
        module_name = module.name

        # Skip modules with underscore (private/internal modules)
        # Check if any part of the module path starts with underscore
        if any(part.startswith("_") for part in module_name.split(".")):
            print(f"Skipping private module: {module_name}")
            continue

        # Skip explicitly excluded modules
        if module_name in excluded_modules:
            print(f"Skipping excluded module: {module_name}")
            continue

        # Parse module path: strands.agent.base -> strands.agent.base.mdx
        parts = module_name.split(".")
        simple_name = parts[-1]
        filename = f"{module_name}.mdx"
        filepath = output_dir / filename
        slug = f"docs/api/python/{module_name}"

        # Render single module
        content = renderer.render_to_string([module])

        content = f"""
---
title: {module_name}
slug:  {slug}
editUrl: false
---
{content}
""".strip()

        if content.strip():  # Only write non-empty files
            # Because we're writing MDX we need to escape brackets so that it's not variable interpolation
            content = content.replace("{", "\\{").replace("<A2A", "&gt;A2A")
            filepath.write_text(content, encoding="utf-8")
            module_files.append((module_name, str(filepath.relative_to(output_dir))))
            print(f"Generated: {filepath}")

    print(f"\nTotal modules documented: {len(module_files)}")


if __name__ == "__main__":
    generate_docs()
