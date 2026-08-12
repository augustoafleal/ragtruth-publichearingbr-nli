# Environment preparation

## Purpose

Create an environment that can run the repository scripts and build this
documentation site.

## Install the project

```bash
python -m pip install -e ".[dev,docs]"
```

## Build the documentation

```bash
mkdocs build --strict
```

## Serve the documentation

```bash
mkdocs serve
```

## Notes

GPU is required for model training and inference workloads. The paired
bootstrap analyses do not require GPU.
