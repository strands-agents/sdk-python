#!/usr/bin/env python3
"""
Setup script for Strands RDS Discovery Tool
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the requirements file
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Get version from environment or default
version = os.getenv("VERSION", "2.0.0")

setup(
    name="strands-rds-discovery",
    version=version,
    author="Strands RDS Discovery Tool Contributors",
    author_email="contributors@example.com",
    description="SQL Server to AWS RDS migration assessment tool with pricing integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/strands-rds-discovery",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/strands-rds-discovery/issues",
        "Documentation": "https://github.com/your-org/strands-rds-discovery/blob/main/README.md",
        "Source Code": "https://github.com/your-org/strands-rds-discovery",
        "Changelog": "https://github.com/your-org/strands-rds-discovery/blob/main/CHANGELOG.md",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "Intended Audience :: Information Technology",
        "Topic :: Database",
        "Topic :: System :: Systems Administration",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: SQL",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pyodbc>=4.0.0",
        "boto3>=1.26.0",
        "pandas>=1.5.0",
    ],
    extras_require={
        "strands": [
            "strands-agents>=1.0.0",
            "strands-agents-tools>=0.2.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "bandit>=1.7.0",
            "safety>=2.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rds-discovery=rds_discovery:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "sql-server",
        "aws",
        "rds",
        "migration",
        "assessment",
        "database",
        "cloud",
        "pricing",
        "compatibility",
    ],
    zip_safe=False,
)
