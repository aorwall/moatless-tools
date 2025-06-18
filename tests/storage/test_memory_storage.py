import asyncio

from moatless.storage.memory_storage import MemoryStorage


def test_basic_operations():
    async def _run():
        storage = MemoryStorage()
        path = "foo/bar.json"
        data = {"hello": "world"}

        await storage.write(path, data)
        assert await storage.exists(path)

        read = await storage.read(path)
        assert read == data

    asyncio.run(_run())


def test_append_and_read_lines():
    async def _run():
        storage = MemoryStorage()
        path = "events.jsonl"

        await storage.append(path, {"id": 1})
        await storage.append(path, {"id": 2})

        lines = await storage.read_lines(path)
        assert lines == [{"id": 1}, {"id": 2}]

    asyncio.run(_run())


def test_list_paths_and_delete():
    async def _run():
        storage = MemoryStorage()
        await storage.write("a/one.json", {"v": 1})
        await storage.write("a/two.json", {"v": 2})

        paths = await storage.list_paths("a")
        assert sorted(paths) == ["a/one.json", "a/two.json"]

        await storage.delete("a/one.json")
        assert not await storage.exists("a/one.json")

    asyncio.run(_run())
