#!/usr/bin/env python
import os

PROJECT_DIRECTORY = os.path.realpath(os.path.curdir)


def remove_file(filepath):
    os.remove(os.path.join(PROJECT_DIRECTORY, filepath))


def main():

    if '{{ cookiecutter.create_author_file }}' != 'y':
        remove_file('AUTHORS.rst')
        remove_file('docs/authors.rst')

    if 'no' in '{{ cookiecutter.command_line_interface|lower }}':
        cli_file = os.path.join('{{ cookiecutter.project_slug }}', 'cli.py')
        remove_file(cli_file)

    if '{{ cookiecutter.use_pypi_with_ci }}' != 'y':
        remove_file('.github/workflows/deploy.yml')

    if '{{ cookiecutter.ci_provider }}' != 'Travis CI':
        remove_file('.travis.yml')
    if '{{ cookiecutter.ci_provider }}' != 'Github Actions':
        remove_file('.github/workflows/docs.yml')
        remove_file('.github/workflows/tests.yml')

    if 'Not open source' == '{{ cookiecutter.open_source_license }}':
        remove_file('LICENSE')


if __name__ == '__main__':
    main()
