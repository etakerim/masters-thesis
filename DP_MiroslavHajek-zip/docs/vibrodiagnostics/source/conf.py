# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# https://stackoverflow.com/questions/2701998/automatically-document-all-modules-recursively-with-sphinx-autodoc
import os
import sys

sys.path.insert(0, os.path.abspath('../../..'))
sys.path.append(os.path.abspath(
    os.path.join(__file__, '../../../../vibrodiagnostics')
))

project = 'Machinery vibrodiagnostics with the IIOT'
copyright = '2024, Miroslav Hájek'
author = 'Miroslav Hájek'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Core library for html generation from docstrings
    'sphinx.ext.autosummary',  # Create neat summary tables
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autodoc_typehints = 'description'

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
