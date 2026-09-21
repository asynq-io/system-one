# CLI

The `system-one` console script. `system-one-mcp` stays as an alias for
`system-one mcp`.

```
system-one fetch  [--variant fp32|int8|fp16] [-o onnx] [--revision main]
system-one export <spec.yaml|spec.json> [-o onnx]
system-one mcp    [--config questions.yaml] [--generic] [--transport stdio]
```

`fetch` needs `pip install "system-one[hub]"`, `export` needs
`pip install "system-one[export]"` (torch); `mcp` needs `system-one[mcp]`.

Both `fetch` and `export` write a `source` block into `<name>.json` recording
the repo, the resolved revision, the weights digest and the graph digest; the
backend checks `graph_sha256` when it loads the graph. See
[`Source`](#system_one.catalog.Source).

::: system_one.catalog.ExportSpec

::: system_one.catalog.Source

::: system_one.catalog.source_block

::: system_one.catalog.load_spec

::: system_one.catalog.fetch_plan

::: system_one.cli.fetch

## Exporter

The `export` subcommand loads `system_one.export`, which imports torch.

::: system_one.export.build_decision_model

::: system_one.export.run_export
