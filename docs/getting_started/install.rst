.. _install:

.. highlight:: shell

Installation
============

Requirements
------------

Python
~~~~~~
**pyteller** has been tested on **GNU/Linux**, and **macOS** systems running `Python 3.6, and 3.7`_ installed.

Also, although it is not strictly required, the usage of a `virtualenv`_ is highly recommended in
order to avoid having conflicts with other software installed in the system where you are trying to run **Orion**.


Install using pip
-----------------

The easiest and recommended way to install **Orion** is using `pip`_:

.. code-block:: console

    pip install pyteller

This will pull and install the latest stable release from `PyPI`_.

Install from source
-------------------

The source code of **Orion** can be downloaded from the `Github repository`_

You can clone the repository and install with the following command in your terminal:

You can clone the repository and install it from source by running ``make install`` on the
``stable`` branch:

.. code-block:: console

    git clone git://github.com/signals-dev/pyteller
    cd pyteller
    git checkout stable
    make install

.. note:: The ``master`` branch of the pyteller repository contains the latest development version. If you want to install the latest stable version, make sure not to omit the ``git checkout stable`` indicated above.

If you are installing **pyteller** in order to modify its code, the installation must be done
from its sources, in the editable mode, and also including some additional dependencies in
order to be able to run the tests and build the documentation. Instructions about this process
can be found in the :ref:`contributing` guide.

.. _Python 3.6, and 3.7: https://docs.python-guide.org/starting/installation/
.. _virtualenv: https://virtualenv.pypa.io/en/latest/
.. _pip: https://pip.pypa.io
.. _PyPI: https://pypi.org/
.. _Github repository: https://github.com/signals-dev/pyteller
