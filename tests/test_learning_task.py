import inspect

import deep_learning_loss
from deep_learning_loss import CrossEntropyLoss, DiceLoss, LearningTask, Loss


def _concrete_losses() -> list[type[Loss]]:
    return [
        member
        for member in vars(deep_learning_loss).values()
        if inspect.isclass(member)
        and issubclass(member, Loss)
        and not inspect.isabstract(member)
        and member is not Loss
    ]


def test_every_task_lists_losses_without_duplicates() -> None:
    for task in LearningTask:
        assert task.losses
        assert len(task.losses) == len(set(task.losses))


def test_every_exported_loss_suits_some_task() -> None:
    losses = _concrete_losses()
    assert len(losses) >= 40
    assert [loss for loss in losses if not LearningTask.find(loss)] == []


def test_every_loss_constructs_with_defaults_and_summarizes_in_one_line() -> None:
    for loss in _concrete_losses():
        loss()
        summary = loss.summarize()
        assert summary and "\n" not in summary and len(summary) <= 120


def test_describe_lists_names_and_summaries() -> None:
    description = LearningTask.SEMANTIC_SEGMENTATION.describe()
    assert description.startswith("SEMANTIC_SEGMENTATION:")
    assert f"- DiceLoss: {DiceLoss.summarize()}" in description


def test_find_returns_tasks_of_a_loss() -> None:
    tasks = LearningTask.find(CrossEntropyLoss)
    assert LearningTask.MULTICLASS_CLASSIFICATION in tasks and LearningTask.SEMANTIC_SEGMENTATION in tasks
