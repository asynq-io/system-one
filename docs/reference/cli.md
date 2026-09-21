# CLI

The `system-one` console script. `system-one-mcp` stays as an alias for
`system-one mcp`.

```
system-one fetch  [--variant fp32|int8|fp16] [-o onnx] [--revision main] [--repo rarha/laya-onnx]
system-one export <spec.yaml|spec.json> [-o onnx]
system-one mcp    [--config questions.yaml] [--generic] [--transport stdio]
```

`fetch` needs `pip install "system-one[hub]"`, `export` needs
`pip install "system-one[export]"` (torch); `mcp` needs `system-one[mcp]`.

::: system_one.cli.ExportSpec

::: system_one.cli.load_spec

::: system_one.cli.fetch_plan

::: system_one.cli.fetch

## Exporter

The `export` subcommand loads `system_one.export`, which imports torch.

::: system_one.export.build_laya

::: system_one.export.run_export
