![example branch parameter](https://github.com/jordanparker6/datascience-starter/workflows/tests/badge.svg?branch=main)
![](https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg?style=flat-square)

# datascience-starter
A starter pack for data science projects.

This repo is designed to bootstrap data science projects by providing a working enviroment for the development of Jupyter Notebooks. Usefull classes from past projects are collected in src/datascience_starter. Theses classes are made available through the datasience_starter package. Please use this classes to aid the development of jupyter notebooks within the notebooks dir.

#### Installation

To install the datascience-starter package, 

1. `git clone https://github.com/jordanparker6/datascience-starter`
2. `cd datascience-starter`
3. `pip install .`

To do... add a Dockerfile or bash one liner to set up.

The datascience-starter package will be available for import within the notebooks using `import datascience_starter`.

#### Use Case

Jupyter Notebooks are the defacto tool for exploring data and presenting work to peers and technical stakeholders. Unforntuately, notebooks are not the best way to re-use code between projects. This is what the datascience-starter repo helps structure. Code snippets for grid search, model diagnosis and data processing (ect.) are not often critical to the notebooks presentation. However, they are often a core step in the analysis pipeline. By abstracting these code snippits to reusable classes, one can develop a reusable codebase and cleaner notebooks.





