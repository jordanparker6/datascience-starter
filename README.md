# datascience-starter
A starter pack for data science projects.

This repo is designed to bootstrap data science projects by providing a working enviroment for the development of Jupyter Notebooks. Usefull classes from past projects are compiled in src/datasceince_starter and are built as a package to aid 
the development of jupyter notebooks within the notebooks dir.

#### Installation

To install the datascience-starter package, 

1. `git clone https://github.com/jordanparker6/datascience-starter`
2. `cd datascience-starter`
3. `pip install .`

The datascience-starter package will be available for import within the notebooks using `import datascience_starter`.

#### Use Case

Jupyter Notebooks are the defacto tool for exploring data and presenting work to peers and stakeholders. Unforntuately, notebook
are not the best way to re-use code between projects. This is what the datascience_starter helps structure. Code snippets for grid search, model diagnosis and data processing (ect.) are not often critical to the notebooks presentation. However, they are often a core step in the analysis pipeline. By abstracting these code snippits to reusable classes, one can develop a re-usable codebase and cleaner notebooks. This is the aim of the datascience-starter.





