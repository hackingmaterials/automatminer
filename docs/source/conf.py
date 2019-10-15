import sys
import os

from docutils.parsers.rst import Directive

import automatminer

# If your extensions are in another directory, add it here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
# General configuration
# ---------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.mathjax',
              'sphinx.ext.autosummary', 'sphinx.ext.coverage',
              'sphinx.ext.napoleon', 'sphinx.ext.intersphinx',
              'sphinx_autodoc_typehints']

autosummary_generate = True

# intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(
        sys.version_info), None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'Automatminer'
copyright = 'HackingMaterials 2019'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

# The short X.Y version.
version = automatminer.__version__

# The full version, including alpha/beta/rc tags.
release = version

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_trees = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Avoid '+DOCTEST...' comments in the docs
trim_doctest_flags = True

# Options for HTML output
# -----------------------

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = '_static/favicon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
    ]
}


# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
html_use_modindex = True

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, the reST sources are included in the HTML build as _sources/<name>.
#html_copy_source = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = ''

# Output file base name for HTML help builder.
htmlhelp_basename = 'ammdoc'


# Options for LaTeX output
# ------------------------

# The paper size ('letter' or 'a4').
#latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
#latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author,
# document class [howto/manual]).
latex_documents = [
  ('index', 'automatminer.tex', 'Automatminer Docs',
   'Alex Dunn', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# Additional stuff for the LaTeX preamble.
#latex_preamble = ''

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_use_modindex = True

html_theme = 'alabaster'

html_theme_options = {
    'logo': 'logo_lowres.png',
    # 'github_repo': 'hackingmaterials/automatminer',
    # 'github_button': 'true',
    # 'link': '#0c7baa',
    # 'show_powered_by': 'false',
    # 'fixed_sidebar': 'true',
    # 'font_family': 'Nunito',
    # 'head_font_family': 'Nunito',
    # 'caption_font_family': 'Nunito',
    # 'description_font_style': 'Nunito',
    # 'description': "An autoML tool for materials",
    # 'font_size': 12,
    # 'code_font_family': 'Ubuntu Mono',
    # 'code_font_size': 16,

    'github_repo': 'hackingmaterials/automatminer',
    'github_button': 'true',
    'link': '#0c7baa',
    'show_powered_by': 'false',
    'fixed_sidebar': 'true',
    'font_family': 'Ubuntu',
    'head_font_family': 'Ubuntu',
    'caption_font_family': 'Ubuntu',
    'description_font_style': 'Ubuntu',
    'description': "An autoML tool for materials",
    'font_size': 11,
    'code_font_family': 'Inconsolata',
    'code_font_size': 12,
    "sidebar_list": "#2b3340",

    # 'page_width': 300,
    'sidebar_width': "300px",
    # "relbarbgcolor": "#333",
    # "sidebarlinkcolor": "#e15617",
    # "sidebarbgcolor": "#000",
    # "sidebartextcolor": "#5685f5",
    # "footerbgcolor": "#111",
    # "linkcolor": "#aa560c",
    # "headtextcolor": "#643200",
    # "codebgcolor": "#f5efe7",
}

# Activate the theme.
# html_theme = 'bootstrap'
# html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# imgmath_image_format = "svg"  # use svg for math
# imgmath_dvisvgm_args = ['--no-fonts', '--exact']
# # imgmath_latex_preamble = '\\usepackage{fouriernc}'
# imgmath_latex_preamble = '\\usepackage{mathptmx}'
# imgmath_font_size = 14
# imgmath_use_preview = True


##############################################################################
# Hack to copy the CHANGES.rst file
# import shutil
#
# try:
#     shutil.copyfile('../../examples/example_settings.yaml',
#                     'example_settings.yaml')
#     # shutil.copyfile('../README.rst', 'README.rst')
# except IOError:
#     pass
#     # This fails during the tesing, as the code is ran in a different
#     # directory

numpydoc_show_class_members = False

suppress_warnings = ['image.nonlocal_uri']


##add custom sphinx exec for loading defaults
import sys
from os.path import basename

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from docutils import nodes, statemachine

class ExecDirective(Directive):
    """Execute the specified python code and insert the output into the document"""
    has_content = True

    def run(self):
        oldStdout, sys.stdout = sys.stdout, StringIO()

        tab_width = self.options.get('tab-width', self.state.document.settings.tab_width)
        source = self.state_machine.input_lines.source(self.lineno - self.state_machine.input_offset - 1)

        try:
            exec('\n'.join(self.content))
            text = sys.stdout.getvalue()
            lines = statemachine.string2lines(text, tab_width, convert_whitespace=True)
            self.state_machine.insert_input(lines, source)
            return []
        except Exception:
            return [nodes.error(None, nodes.paragraph(text = "Unable to execute python code at %s:%d:" % (basename(source), self.lineno)), nodes.paragraph(text = str(sys.exc_info()[1])))]
        finally:
            sys.stdout = oldStdout

def setup(app):
    app.add_directive('exec', ExecDirective)
