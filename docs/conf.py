"""Sphinx configuration for the sampleworks documentation."""

project = "sampleworks"
author = "Prism Science"

# MyST renders the Markdown; autodoc and napoleon turn the NumPy-style docstrings in
# src/sampleworks into API pages. Modules whose imports are missing on the runner emit a
# warning and are skipped, which is why the build is not run with -W.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]

exclude_patterns = ["_build"]

html_theme = "furo"
