# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'probeinterface'
copyright = '2020, Samuel Garcia'
author = 'Samuel Garcia'

# This a hack to get the version inside the code
exec(open("../probeinterface/version.py").read())


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx_gallery.gen_gallery',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']





from sphinx_gallery.sorting import ExplicitOrder
from sphinx_gallery.sorting import FileNameSortKey

# for sphinx gallery plugin
sphinx_gallery_conf = {
    'examples_dirs': ['../examples', ],   # path to your example scripts
    'gallery_dirs': ['examples'],  # path where to save gallery generated examples
    'subsection_order': ExplicitOrder([
                                       ]),
    #~ 'within_subsection_order': FileNameSortKey,
    'ignore_pattern': '/generate_',
    'filename_pattern' : 'ex_',
    'download_all_examples': False,
}