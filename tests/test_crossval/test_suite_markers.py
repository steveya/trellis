"""Architecture checks for the cross-validation suite."""

from __future__ import annotations


def test_crossval_suite_is_marked_crossval(request):
    assert request.node.get_closest_marker("crossval") is not None
