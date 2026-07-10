"""Sphinx configuration for the RecourseBench documentation site."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Make the repository root importable so autodoc can find the top-level
# packages (dataset, model, method, evaluation, preprocess, utils, experiments).
sys.path.insert(0, os.path.abspath(".."))

project = "RecourseBench"
author = "RecourseBench authors"
copyright = f"{datetime.now():%Y}, {author}"

try:
    import importlib.metadata as _im

    release = _im.version("recourse_bench")
except Exception:  # pragma: no cover - docs build without install
    release = "0.1.0"
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "class"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__getitem__,__len__,__call__",
}

# Concrete components pull heavy / optional solver dependencies. The reference
# documents the base classes, so mock the optional backends to keep the build
# fast and portable.
autodoc_mock_imports = [
    # Heavy runtime deps not needed to render the base-class docstrings.
    "torch",
    "torchvision",
    "sklearn",
    "scipy",
    "tqdm",
    # Method-specific solver / optional backends.
    "gurobipy",
    "z3",
    "clingo",
    "cvxpy",
    "alibi",
    "lime",
    "art",
    "dice_ml",
    "pysmt",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- HTML output (numpy.org-style pydata theme) ------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"RecourseBench {version}"
html_static_path = ["_static"]
html_sidebars = {
    "**": [],
}
html_theme_options = {
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/charmlab/recourse_bench",
            "icon": "fa-brands fa-github",
        }
    ],
}
