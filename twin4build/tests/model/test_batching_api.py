from twin4build.model.model import Model


def test_batching_api_is_hard_renamed():
    assert callable(Model.batch_components)
    assert callable(Model.get_batched_component_info)
    assert callable(Model.get_batch_id_for_component)
    assert not hasattr(Model, "build_compiled_model")
    assert not hasattr(Model, "get_compiled_component_info")
    assert not hasattr(Model, "get_block_id_for_component")
