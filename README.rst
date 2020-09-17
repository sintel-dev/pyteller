======================
Cookiecutter PyPackage
======================

Cookiecutter_ template for a Python package, customized for MIT DAI Lab.

This is a slight fork of https://github.com/audreyr/cookiecutter-pypackage.

Why do this stuff? See `Why???`_.

For any bugs, open an Issue.

* GitHub repo: https://github.com/DAI-Lab/cookiecutter-pypackage/
* Free software: BSD license

Features
--------

* Testing setup with ``unittest`` and ``pytest``
* Github-Actions_: Ready for Github Actions Continuous Integration testing
* Travis-CI_: Ready for Travis Continuous Integration testing
* Tox_ testing: Setup to easily test for Python 2.7, 3.5, 3.6, 3.7, 3.8
* Sphinx_ docs: Documentation ready for generation, automatic building and deploying to gh-pages (strongly recommended)
* Bumpversion_: Pre-configured version bumping with a single command
* Auto-release to PyPI_ when you push a new tag to master (strongly recommended)
* Command line interface using Click (optional)

Quickstart
----------

Install the latest Cookiecutter if you haven't installed it yet (this requires
Cookiecutter 1.4.0 or higher)::

    pip install cookiecutter

Generate a Python package project (this will create a new subdirectory of your
current working directory)::

    cookiecutter gh:DAI-Lab/cookiecutter-pypackage

Then:

1. Initialize a git repo in your new project

   * Commit the new project (``git add .`` and ``git commit -m "Initial commit"``)

2. Create and test your development environment

   * Create a virtual environment of your choice ( e.g. ``conda create -n project_name_env python=3.6``, ``virtualenv``, ``pyenv``, etc.).
   * Install the dev environment (``make install-develop``).
   * Check that the tests work (``make test``)
   * Check that the linter works (``make lint``)

3. Make sure that you can test your project using tox

   * Deactivate any virtual environments. (This may or may not be necessary for you, tox consistenly does *not* work inside conda.)
   * Install tox (``pip install tox``)
   * Run tox (``tox``)

4. Configure Travis to run tests and build documentation. If you're using Github Actions, you can skip this step.


   * Create the repo on GitHub if it is not already there
   * Add the repo to your Travis-CI_ account (https://travis-ci.org/ORG_NAME/PROJECT_NAME)
   * Push a commit and monitor the build

5. Configure Travis to automatically deploy documentation to GitHub pages. If you're using Github Actions, you can skip this step.

   * Configure a GitHub deployment token and add it to Travis. See the ``.travis.yml`` file of your project for details.
   * Push a commit and watch your documentation being built.
   * View the documentation at https://hdi-project.github.io/PROJECT_NAME

6. Automatically deploy to PyPI on new tags.

   * Register_ your project with PyPI, or use the DAI Lab account (`dai_lab_mit`, ask someone for the info)
   * Use Travis to encrypt your PyPI password in Travis config and activate automated deployment on PyPI when you push a new tag to master branch - see the ``.travis.yml`` file of your project for details. Alternatively, if using Github Actions, you can generate an API token on PyPI and add it to your Github Secrets with the name ``pypi_password`` - see the ``.github/workflows/deploy.yml`` file of your project for details.
   * Test the release process

       * Ensure you can create the dist using ``make dist``
       * `Sign up for a Test PyPI account`_ (this is different from normal PyPI but looks exactly the same)
       * Ensure that ``make test-release`` works
       * View your test release at https://test.pypi.org/project/PROJECT_NAME (or similar)

   * Tag a new release using ``bumpversion`` (TODO)
   * Push the latest *commit* to master and ensure that tests pass (``git push origin master``)
   * Push the latest *tag* to master and watch as Travis / Github Actions will automatically deploy your release to PyPI (``git push --tags origin master``)

7. Start developing!

For more details, see the `cookiecutter-pypackage tutorial`_. Note that some aspects may not be relevant to the DAI Lab fork.

Why???
------

Why should you use this? Why do you want these features? Here are some quick and
dirty answers, that will hopefully get expanded and referenced with appropriate
links.

You're probably doing some of these things already, like structuring your python
package in a standard manner, writing README, a setup.py, and writing tests.
This is not enough to (a) distribute your code (b) get people to trust your code
(c) get people to download your code (d) get people to use your code.

Let this template handle the little things. You will avoid the following:

* committing ``.pyc`` files or other binary files because you never thought to add
  a ``.gitignore``
* omitting a ``LICENSE`` which means that other people cannot legally reuse your
  software
* having your tests pass locally but finding that a stranger on the internet
  can't install your code because they use Python 3.5 and you use Python 3.6
* having a sequence of 8 commits (``add Travis``, ``see if this fix makes Travis
  work now``, ``Travis still doesn't work``, etc.) because you are rolling your own
  configuration for different services one project at a time
* spending time and effort figuring out how to release your project on PyPI
* releasing your project on PyPI and finding that people can't install it
  successfully because you misconfigured one release
* etc.

Distribution
~~~~~~~~~~~~

Do you want people to use your software? If so

* you need to make it available on PyPI
* you need to make it trustworthy with up-to-date testing and documentation that
  can be verified and referenced
* you need to signal that it is high-quality by following best practices

It is imperative that ``pip install your_package`` works out of the box for as
many people as possible, and that they can quickly assess whether your software
is high-quality and deserves a chance to be used.

Testing
~~~~~~~

If you're reading this, you should be testing your code. With unittests and
integration tests if applicable. Full stop.

There are several unit testing frameworks in Python. You can use whatever one
you want. Unittest in Python 3 is part of the standard library and is pretty
solid. You can use pytest as a test runner. It's easiest if we are consistent
across the lab.

But writing and running your tests is just one part of the picture. ``tox`` makes
sure that your code works/your tests pass on every version of Python you claim
to support and all other dependencies your require. What is the probability that
the person who wants to pip install your package is using the same version of
Python as you are? This is standard in python projects.

Finally, you should run your tests automatically everytime you update your
software using continuous integration.

Docs
~~~~

You should document your code.

It is not enough to document your code in docstrings. Users across the world
should be able to quickly view your documentation on the web.

It is not enough to have your documentation on the web. Built documentation
should be standardized across the lab with a consistent theme/style to look
polished.

It is not enough to have a standardized look and feel. Built documentation
should be automatically updated to reflect the current state of your repository.

Finally, it is not enough to just write docstrings etc. You should also write
expository documentation: introduction, installation, quick start/basic usage,
tutorial/advanced usage, examples, faq, how to contribute, API reference.

Dev tools
~~~~~~~~

Use dev tools to make your life easier.

Want to release a new version of your software?

1. Update your HISTORY file
2. Use ``bumpversion`` to tag a new version following semantic versioning.
3. Push your commit and tags to GitHub, and have your CI service automatically
   deploy a new release to PyPI.

Want to automatically build and deploy your documentation?

1. Literally, do nothing different. If you configure your CI provider
   correctly, your documentation can be rebuild and redeployed on every commit.

Glossary (for newbs)
--------------------

You're about to add all the things to your repo that make your repo a legitimate
open-source python project that other developers will look at and say "wow, that
is a legit looking python project". The things you're going to add make it easy
for other developers to understand the work you did, for you to test your code,
for legal issues to be avoided, for easy pushing to public python repos. The
things are files and folders and I'll give you a quick overview of what they
are/do.

* Tox_ (tox.ini): A system that can run all kinds of tests for you. For
  instance, you can test your code on various versions (Python 2.7, 3.5,
  3.6, 3.7, 3.8) and test your code on linters as well.

* Travis-CI_ (travis.yml): A continuous integration system. That means every
  time you push a commit it will simulate downloading your project, installing
  dependencies and running all your tests, to ensure your project is continously
  up to specification.

* README (README.md): A file that people should read if they want to understand
  your project!

* reStructuredText: A markup language that is often used for writing
  documentation in Python projects. It is more powerful than Markdown but a bit
  harder to learn. See `reStructuredText reference`_.

* setup.py: a file that contains configuration info for installing your project.
  Among many other things, our version of setup.py includes lists of
  dependencies for people who want to run the tests and dependencies developers
  who want to hack on the project, build documentation, and create new releases.
  See setup.py_.

* setup.cfg: This is a general configuration file that can be read by all sorts
  of development tools, including ``setuptools``, ``flake8``, ``isort``,
  ``bumpversion``, and more. It uses the `INI file format`_.

* PyPI_: A website that hosts and allows for easy install of python libraries
  (this is where pip install downloads from). PyPI and the community of python
  developers make python the awesome and flexible language that you know.

* AUTHORS.rst: A list of authors.

* CONTRIBUTING.rst: Information for people interested in contributing bug
  reports, new features, documentation, and more.

* HISTORY.md: A list of previous releases, including new features introduced and
  breaking changes.

* LICENSE: A legal license that explains how the code can legally be used.
  (Typically we use MIT's open source license).

* Makefile: A helpful file with pre-defined bash commands. Try running ``make
  help`` to see the list of commands, including ones that will run your tests
  for you.

* MANIFEST.in: Loosely, this file lists non-Python source files that should be
  included in the distribution you upload to PyPI. See Manifest.in_.

* .gitignore: Ignore files matching the regex patterns defined in here.
  (*A.k.a.* a good way to avoid committing log files or pyc files etc.)

.. _`pip docs for requirements files`: https://pip.pypa.io/en/stable/user_guide/#requirements-files
.. _`Sign up for a Test PyPI account`: https://test.pypi.org/account/register/
.. _Register: https://packaging.python.org/distributing/#register-your-project
.. _`cookiecutter-pypackage tutorial`: https://cookiecutter-pypackage.readthedocs.io/en/latest/tutorial.html
.. _Cookiecutter: https://github.com/DAI-Lab/cookiecutter
.. _Travis-CI: http://travis-ci.org/
.. _Github-Actions: https://github.com/features/actions
.. _Tox: http://testrun.org/tox/
.. _Sphinx: http://sphinx-doc.org/
.. _Bumpversion: https://github.com/peritus/bumpversion
.. _PyPI: https://pypi.python.org/pypi
.. _`INI file format`: https://en.wikipedia.org/wiki/INI_file
.. _`setup.py`: https://packaging.python.org/tutorials/distributing-packages/#setup-py
.. _`MANIFEST.in`: https://packaging.python.org/tutorials/distributing-packages/#manifest-in
.. _`reStructuredText reference`: https://gist.github.com/ionelmc/e876b73e2001acd2140f
