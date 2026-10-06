# Contributing

## Comments and docstrings

Code is read far more often than it is written. Keep the prose in it about what the code does now.

- **Docstrings state the current spec** of each module, class, function and method: what it does, what it takes and returns, what `None` or an empty result means, and what it raises. Start with a one-line summary. Add a body only when the contract needs one.
- **Comments explain only what the code can't.** Use one where something non-obvious is happening, or where a reader needs context the code doesn't carry. If the code already says it, don't repeat it.
- **No history in code.** Don't describe how something used to work, which bug a line fixed, or where a function was factored out of. That belongs in the commit message and the issue.
- **Design decisions and measurements go in `docs/`.** If the reasoning behind a threshold, rule or structure is worth keeping, write it up under [`docs/decisions/`](docs/decisions/), headed by the name of the code it explains. The same goes for an evaluation or a timing that justifies a choice. Leave at most a one-line pointer in the code.
- **Issue numbers stay out of code**, unless the issue is the only place a constraint that still holds is explained.

## Checks

Run before opening a pull request:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
```

`pre-commit install` runs ruff and the file hygiene hooks on each commit.
