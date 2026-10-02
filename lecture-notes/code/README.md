# Scientific Machine Learning — code

Code and notebooks for the course, and the interpreter sessions printed in the
lecture notes *Scientific Machine Learning*.

| Folder | Contents |
|---|---|
| `notebooks/13-arrays-vectorization-memory` | arrays, vectorization, broadcasting, measuring (Chapter 13) |
| `notebooks/14-concurrency-compilation` | threads, processes, shared memory, numba (Chapter 14) |
| `notebooks/15-frameworks` | PyTorch, JAX, and Flax (Chapters 3 and 15) |
| `code/` | modules imported by the notebooks |
| `examples/` | the interpreter sessions included in the notes |

## Setup

Poetry creates the virtual environment inside this folder (`.venv/`, see
`poetry.toml`), so VS Code finds the notebook kernel automatically.

```bash
poetry install --with dev
```

## Checking the sessions printed in the notes

Every session in `examples/` is a doctest: the output printed in the notes is
the output Python produces. Run them all with

```bash
poetry run pytest
```

The comments `#<1>`, `#<2>`, ... in the sessions mark the circled numbers that
the notes explain under each listing.

## Running the notebooks

```bash
poetry run jupyter lab
```

or open a notebook in VS Code and choose the `.venv` kernel.
