import pytest

from perception.nodes.scheduled_camera_detector_node import _camera_specs


def test_camera_specs_scale_to_twelve_camera_registry():
    ids = [f"camera_{letter}" for letter in "ABCDEFGHIJKL"]
    models = ["external_camera", *[f"external_camera_{letter.lower()}" for letter in "BCDEFGHIJKL"]]
    topics = [f"/{model}/image_raw" for model in models]

    assert _camera_specs(ids, models, topics) == list(zip(ids, models, topics))


@pytest.mark.parametrize(
    "ids,models,topics",
    [
        (["camera_A"], ["external_camera"], []),
        (["camera_A", "camera_A"], ["external_camera", "external_camera_b"], ["/a", "/b"]),
        (["camera_A", "camera_B"], ["external_camera", "external_camera"], ["/a", "/b"]),
        (["camera_A", "camera_B"], ["external_camera", "external_camera_b"], ["/a", "/a"]),
    ],
)
def test_camera_specs_reject_misaligned_or_ambiguous_registries(ids, models, topics):
    with pytest.raises(ValueError):
        _camera_specs(ids, models, topics)
