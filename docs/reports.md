# Safety Reports

Each metric in `report.json` and `report.html` includes:

- observed peak value;
- configured limit;
- ratio to the limit;
- unit;
- evaluation status;
- peak time and electrode when available;
- the rationale for the configured boundary.

Metric statuses are:

- `within_limit`
- `exceeds_limit`
- `not_evaluated`

Overall statuses are:

- `within_configured_limits`
- `configured_limits_exceeded`
- `incomplete_evaluation`

An incomplete report usually means an optional metric such as CEM43 was not
enabled or the raw data did not contain the required series.

These statuses compare model outputs with configured research limits. They do
not certify a device or protocol as clinically safe.

Regenerate a report from existing raw data with:

```bash
dynaphos report results/my_run
```
