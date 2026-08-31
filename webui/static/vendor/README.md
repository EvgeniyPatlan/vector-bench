# Vendored front-end dependencies

Committed rather than fetched from a CDN: a benchmark host often has no network,
and a chart library that silently fails to load would leave the explorer blank
with no explanation.

| File | Project | Version | Licence |
| --- | --- | --- | --- |
| `uPlot.iife.min.js`, `uPlot.min.css` | [uPlot](https://github.com/leeoniya/uPlot) | 1.6.31 | MIT |

To update, replace both files from
`https://cdn.jsdelivr.net/npm/uplot@<version>/dist/` and change the version
here. Nothing else references the version.
