# Baseball Sim Engine

Quickstart:
```bash
make setup
make test
make serve  # http://localhost:8000/health
```

See `docs/BUILD_PLAN.md` for step-by-step tasks, `docs/RATINGS_MAPPING.md` for how your ratings are used, and `docs/API_REFERENCE.md` for endpoint contracts.

Data ETL & Training:
- See `docs/RETROSHEET_ETL.md` for converting Retrosheet data, building the PA table, and training the PA model artifact that the server autoloads.

Data Sources & Licensing:
- See `docs/DATA_SOURCES.md` for required Retrosheet attribution and links to other sources and tools used.
