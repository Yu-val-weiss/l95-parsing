# L95 Parsing

This repository contains code for dependency and constituency parsing using the Stanza and Supar libraries.
This repository was written for L95's final project.

## Requirements

- Python 3.12
- Poetry for dependency management

### Poetry

Information regarding the installation of Poetry can be found [here](https://python-poetry.org/docs/#installation).

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Yu-val-weiss/l95-parsing.git
cd l95-parsing
```

2. Install dependencies using Poetry:

```bash
poetry install
```

## Usage

The project provides two main parsing capabilities:

1. Dependency Parsing (using Stanza)
2. Constituency Parsing (using Supar)

### Running the Parser

Use the CLI interface:

```bash
poetry run cli [OPTIONS] COMMAND [ARGS]
```

All necessary models will be automatically downloaded.

To get information about the CLI run the following:

```bash
poetry run cli --help
```

Can also run each of the options (`evaluate`, `predict` and `visualise`) directly with

```bash
poetry run evaluate
poetry run predict
poetry run visualise
```

## Project Structure

```txt
src/
├── task/                   # Core parsing functionality
│   ├── eval/               # Evaluation metrics and scoring
│   │   ├── constituency.py # Constituency parsing evaluation
│   │   ├── dep_rel.py      # Dependency parsing evaluation
│   │   └── score.py        # Scoring utilities
│   └── predict.py          # Main parsing implementations
├── utils/                  # Utility functions
│   ├── conllu.py           # Conllu format utils
│   ├── constituency.py     # Constituency parser utils
│   ├── dep_rel.py          # Dependence relation utils
│   ├── parse_task_file.py  # File parsing utilities
│   ├── stan2uni.ini        # Contains rules for converting Stanford dependence tags to UD
│   ├── stanza.py           # Stanza-specific utilities  
│   └── task_data.py        # Task data handling functions
└── main.py                 # CLI for prediction/evaluation/visualisation.

prediction_files/           # Input (task) files
├── constituencies.txt      # Predicted constituency parses
└── dep_rel.txt             # Predicted dependency relations

task_files/                 # Input (task) files
├── sentences.txt           # Input sentences
├── constituencies.txt      # Gold constituency parses
└── dep_rel.txt             # Gold dependency relations
```
