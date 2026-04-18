"""Architecture checks for the task-challenge suite."""


def test_task_suite_is_marked_task_challenge(request):
    assert request.node.get_closest_marker("task_challenge") is not None
