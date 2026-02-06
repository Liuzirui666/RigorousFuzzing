# README


## Structure

The structure of our code is: 

```text
fuzz
├── experiment-config.yaml
└── fuzzbench
    ├── alembic.ini
    ├── .allstar
    ├── analysis
    ├── benchmarks
    ├── common
    ├── compose
    ├── config
    ├── conftest.py
    ├── CONTRIBUTING.md
    ├── database
    ├── docker
    ├── .dockerignore
    ├── docs
    ├── experiment
    ├── fuzzbench
    ├── fuzzers
    ├── .gcloudignore
    ├── .github
    ├── .gitignore
    ├── .gitmodules
    ├── LICENSE
    ├── Makefile
    ├── presubmit.py
    ├── .pylintrc
    ├── .pytest_cache
    ├── pytest.ini
    ├── README.md
    ├── requirements.txt
    ├── service
    ├── split
    │   ├── plots
    │   ├── run_multiple.py
    │   ├── run_split_union.py
    │   ├── seed_utils.py
    │   ├── sparsity.py
    │   └── split_strategy_multi.py
    ├── src_analysis
    ├── .style.yapf
    ├── test_libs
    └── third_party

```

## Prerequisites

You will need to setup [Fuzzbench](https://google.github.io/fuzzbench/getting-started/prerequisites/) environment first in order to run local experiments.

## Running Experiments

Setting up the experiments properly within the split folder, and then run
```bash
PYTHONPATH=. python3 fuzz/fuzzbench/split/run_multiple.py
```

