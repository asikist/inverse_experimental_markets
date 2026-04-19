"""
Minimal utilities used by the gradient_boosting notebooks.

The full original module (with torch padding/tensor helpers and other
unused machinery) is preserved at ``extra_code/models/utilities.py``;
only ``dw`` / ``DummyWriter`` is imported from this module by the code
on the reproduction path.
"""


class DummyWriter:
    """No-op writer used as a placeholder log sink in the notebooks."""

    def write(self, msg):
        pass


dw = DummyWriter()
