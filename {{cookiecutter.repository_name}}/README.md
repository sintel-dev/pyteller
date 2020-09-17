<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<!-- Uncomment these lines after releasing the package to PyPI for version and downloads badges -->
<!--[![PyPI Shield](https://img.shields.io/pypi/v/{{ cookiecutter.package_name }}.svg)](https://pypi.python.org/pypi/{{ cookiecutter.package_name }})-->
<!--[![Downloads](https://pepy.tech/badge/{{ cookiecutter.package_name }})](https://pepy.tech/project/{{ cookiecutter.package_name }})-->
{%- if cookiecutter.ci_provider == 'Travis CI' %}
[![Travis CI Shield](https://travis-ci.org/{{ cookiecutter.github_owner }}/{{ cookiecutter.repository_name }}.svg?branch=master)](https://travis-ci.org/{{ cookiecutter.github_owner }}/{{ cookiecutter.repository_name }})
{%- endif %}
{%- if cookiecutter.ci_provider == 'Github Actions' %}
[![Github Actions Shield](https://img.shields.io/github/workflow/status/{{ cookiecutter.github_owner }}/{{ cookiecutter.repository_name }}/Run%20Tests)](https://github.com/{{ cookiecutter.github_owner }}/{{ cookiecutter.repository_name }}/actions)
{%- endif %}
{%- if cookiecutter.use_codecov_with_ci == 'y' %}
[![Coverage Status](https://codecov.io/gh/{{ cookiecutter.github_owner }}/{{ cookiecutter.repository_name }}/branch/master/graph/badge.svg)](https://codecov.io/gh/{{ cookiecutter.github_owner }}/{{ cookiecutter.repository_name }})
{%- endif %}



# {{ cookiecutter.project_name }}

{{ cookiecutter.project_short_description }}

{% if cookiecutter.open_source_license == 'Not open source' -%}
- Free software: {{ cookiecutter.open_source_license }}
{% endif -%}
- Documentation: https://{{ cookiecutter.github_owner }}.github.io/{{ cookiecutter.repository_name }}
- Homepage: https://github.com/{{ cookiecutter.github_owner }}/{{ cookiecutter.repository_name }}

# Overview

TODO: Provide a short overview of the project here.

# Install

## Requirements

**{{ cookiecutter.project_name }}** has been developed and tested on [Python {% if cookiecutter.support_py2 == 'y' %}2.7, {% endif %}3.5, 3.6, 3.7 and 3.8](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/)
is highly recommended in order to avoid interfering with other software installed in the system
in which **{{ cookiecutter.project_name }}** is run.

These are the minimum commands needed to create a virtualenv using python3.6 for **{{ cookiecutter.project_name }}**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) {{ cookiecutter.repository_name }}-venv
```

Afterwards, you have to execute this command to activate the virtualenv:

```bash
source {{ cookiecutter.repository_name }}-venv/bin/activate
```

Remember to execute it every time you start a new console to work on **{{ cookiecutter.project_name }}**!

<!-- Uncomment this section after releasing the package to PyPI for installation instructions
## Install from PyPI

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **{{ cookiecutter.project_name }}**:

```bash
pip install {{ cookiecutter.package_name }}
```

This will pull and install the latest stable release from [PyPI](https://pypi.org/).
-->

## Install from source

With your virtualenv activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:{{ cookiecutter.github_owner }}/{{ cookiecutter.repository_name }}.git
cd {{ cookiecutter.repository_name }}
git checkout stable
make install
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

Please head to the [Contributing Guide](https://{{ cookiecutter.github_owner }}.github.io/{{ cookiecutter.repository_name }}/contributing.html#get-started)
for more details about this process.

# Quickstart

In this short tutorial we will guide you through a series of steps that will help you
getting started with **{{ cookiecutter.project_name }}**.

TODO: Create a step by step guide here.

# What's next?

For more details about **{{ cookiecutter.project_name }}** and all its possibilities
and features, please check the [documentation site](
https://{{ cookiecutter.github_owner }}.github.io/{{ cookiecutter.repository_name }}/).
