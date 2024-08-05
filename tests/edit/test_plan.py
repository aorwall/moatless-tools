from moatless.edit.plan import ApplyChange


def test_deserialize_action():
    data = {
        "scratch_pad": "To fix the race condition in the has_key method, we need to handle the case where the file might be deleted between the os.path.exists check and the open call. We can do this by wrapping the open call in a try-except block to catch the FileNotFoundError exception and return False if the file is not found.",
        "action": "modify",
        "instructions": "Wrap the open call in the has_key method in a try-except block to catch FileNotFoundError and return False if the file is not found.",
        "file_path": "django/core/cache/backends/filebased.py",
        "span_id": "FileBasedCache.has_key",
    }

    action = ApplyChange.model_validate(data)

    assert action.scratch_pad == data["scratch_pad"]
    assert action.action == data["action"]
    assert action.instructions == data["instructions"]
    assert action.file_path == data["file_path"]

