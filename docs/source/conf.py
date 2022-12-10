# Path imports for sphinx.
import sys, pathlib, os

main_dir = pathlib.Path(".").resolve()

models_dir = pathlib.Path(f"{main_dir}/model".replace("\\docs\\source","")).resolve()
api_dir = pathlib.Path(f"{main_dir}/API".replace("\\docs\\source","")).resolve()
gui_dir = pathlib.Path(f"{main_dir}/GUI".replace("\\docs\\source","")).resolve()

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, str(api_dir))
sys.path.insert(0, str(gui_dir))
sys.path.insert(0, str(models_dir))

print(sys.path)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RL-KG'
copyright = '2022, DEAL'
author = 'DEAL'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
