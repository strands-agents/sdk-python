"""Tests for project configuration consistency."""

from pathlib import Path

import tomli


def test_optional_dependencies_version_consistency():
    """Test that duplicate dependencies across groups have consistent version specifiers."""
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)

    optional_deps = data["project"]["optional-dependencies"]

    # Collect all dependencies and their version specifiers with group info
    dep_specs: dict[str, tuple[str, str]] = {}

    for group_name, deps in optional_deps.items():
        for dep in deps:
            # Extract package name before any version specifier
            name = dep
            for op in [">=", "==", ">", "<", "~=", "!="]:
                if op in name:
                    name = name.split(op)[0].strip()
                    break

            # Remove extras like [sql] from name
            if "[" in name:
                name = name.split("[")[0].strip()

            # Extract version specifier (everything after package name)
            version_spec = dep[len(name) :].strip()
            # Remove extras from version spec if present
            if version_spec.startswith("["):
                bracket_end = version_spec.find("]")
                if bracket_end != -1:
                    version_spec = version_spec[bracket_end + 1 :].strip()

            if name in dep_specs:
                (previous_spec, first_group) = dep_specs[name]
                assert previous_spec == version_spec, (
                    f"Version specifier mismatch for {name}: '{dep_specs[name]}'"
                    f" in [{first_group}] vs '{version_spec}' in [{group_name}]"
                )
            else:
                dep_specs[name] = (version_spec, group_name)
