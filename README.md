![example branch parameter](https://github.com/jordanparker6/datascience-starter/workflows/tests/badge.svg?branch=main)
![](https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg?style=flat-square)

# datascience-starter
A starter pack for data science projects.

This repo is designed to bootstrap data science projects by providing a working enviroment for the development of Jupyter Notebooks. Usefull classes from past projects are collected in src/datascience_starter. Theses classes are made available through the datasience_starter package. Please use this classes to aid the development of jupyter notebooks within the notebooks dir.

#### Use Case

Jupyter Notebooks are the defacto tool for exploring data and presenting work to peers and technical stakeholders. Unforntuately, notebooks are not the best way to re-use code between projects. This is what the datascience-starter repo helps structure. Code snippets for gridsearch, model diagnosis and data processing (ect.) are not often critical to the notebooks presentation. However, they are often a core step in the analysis pipeline. By abstracting these code snippits to reusable classes, one can develop a reusable codebase and cleaner notebooks.

#### Installation

To install the datascience-starter package, 

1. `git clone https://github.com/jordanparker6/datascience-starter`
2. `cd datascience-starter`
3. `pip install .` or `make install`

### Usage

The datascience-starter package can be used in any python file or jupyter notebook once installed using `import datascience_starter`. To take advantage of the project structure, all project working should be completed in the notebooks directory and notebooks should harness the reusable code maintained in the datascience-starter package.

Other usage comands are as follows.

1. To build the documentation for the datascience-starter package `make docs`.
2. To run tests over the datasceince-starter package `make tests`.
3. To run linting over the datascience-starter package `make lint`.





