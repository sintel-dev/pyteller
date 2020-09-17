.. include:: readme.rst

.. toctree::
   :hidden:
   :maxdepth: 2

   Overview <readme>

.. toctree::
   :caption: Resources
   :hidden:

   API Reference <api/{{ cookiecutter.project_slug }}>
   contributing
   {% if cookiecutter.create_author_file == 'y' -%}
   authors
   {% endif -%}
   history

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
