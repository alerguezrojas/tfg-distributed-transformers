"""Datasets: one ``DatasetSpec`` (``src.data.spec``) per dataset, looked up by name
through ``src.data.registry``.

Kept import-free on purpose: light consumers (the web dashboard) import pure
constants such as ``src.data.bigearthnet_classes`` without pulling in torch.
"""
