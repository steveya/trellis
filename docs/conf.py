# Sphinx configuration for Trellis documentation

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'Trellis'
copyright = '2024-2026, Steve Yang'
author = 'Steve Yang'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
}

# -- Extension configuration -------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# MathJax for LaTeX rendering
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# Source suffix
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Suppress warnings for missing imports in autodoc
autodoc_mock_imports = ['anthropic', 'openai', 'fredapi', 'requests', 'blpapi']
