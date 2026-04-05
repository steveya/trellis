"""Architecture checks for the verification suite."""

from __future__ import annotations


def test_verification_suite_is_marked_verification(request):
    assert request.node.get_closest_marker("verification") is not None
