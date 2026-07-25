"""ProgressBus: async pub/sub, per-job channels."""
import asyncio
import pytest
from b2cq.progress import ProgressBus


@pytest.mark.asyncio
async def test_subscribe_receives_published_events():
    bus = ProgressBus()
    received = []
    async def consumer():
        async for evt in bus.subscribe("job1"):
            received.append(evt)
            if len(received) == 2:
                break
    consumer_task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)
    await bus.publish("job1", {"type": "log", "msg": "hello"})
    await bus.publish("job1", {"type": "status", "quant": "Q4_K_M", "status": "running"})
    await consumer_task
    assert received == [
        {"type": "log", "msg": "hello"},
        {"type": "status", "quant": "Q4_K_M", "status": "running"},
    ]


@pytest.mark.asyncio
async def test_channels_isolated():
    bus = ProgressBus()
    got = []
    async def consumer():
        async for evt in bus.subscribe("jobA"):
            got.append(evt); break
    t = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)
    await bus.publish("jobB", {"m": "wrong"})
    await bus.publish("jobA", {"m": "right"})
    await t
    assert got == [{"m": "right"}]


@pytest.mark.asyncio
async def test_subscriber_queue_removed_after_consumer_exits():
    # Plain `break` out of `async for` does NOT synchronously close the
    # underlying async generator (its `finally` only runs on GC, which is
    # non-deterministic), so cancel the consumer task instead — cancellation
    # delivers CancelledError at the generator's suspended `await q.get()`,
    # which runs the `finally` cleanup synchronously as part of unwinding.
    bus = ProgressBus()
    received = []

    async def consumer():
        async for evt in bus.subscribe("job1"):
            received.append(evt)

    consumer_task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)
    await bus.publish("job1", {"type": "log", "msg": "hello"})
    await asyncio.sleep(0.01)

    consumer_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer_task

    assert received == [{"type": "log", "msg": "hello"}]
    assert bus._queues["job1"] == []
