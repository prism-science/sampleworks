"""Sphinx configuration for the sampleworks documentation."""

project = "sampleworks"
author = "Prism Science"

# MyST renders the Markdown; autoapi and napoleon turn the NumPy-style docstrings in
# src/sampleworks into API pages. autoapi parses the source instead of importing it, so
# the model wrappers document fine even though each one's dependencies live in a
# separate pixi environment.
extensions = [
    "autoapi.extension",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autoapi_dirs = ["../src/sampleworks"]
autoapi_options = ["members", "undoc-members", "show-inheritance", "show-module-summary"]
# Render "Attributes" sections as fields so they do not duplicate the attributes autoapi
# already documents.
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "biotite": ("https://www.biotite-python.org/latest", None),
}

myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]

exclude_patterns = ["_build"]

html_theme = "furo"
