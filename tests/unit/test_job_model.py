"""Job / QuantResult / JobStore basics."""
from b2cq.job_model import Job, QuantResult, QuantStatus, JobStore
from b2cq.calibration import CalibrationSource


def test_jobstore_create_and_get():
    store = JobStore()
    job = store.create(
        source_model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        owner="bigblueceiling",
        quants=[QuantResult(quant_id="Q4_K_M", status=QuantStatus.PENDING, lane="B")],
        calibration=CalibrationSource(type="bundled"),
        private=False,
        update_source_readme=True,
    )
    assert store.get(job.id) is job
    assert store.list() == [job]
