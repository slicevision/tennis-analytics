"""Allow ``python -m src`` to invoke the pipeline CLI."""

from .pipeline import main

if __name__ == "__main__":
    main()
