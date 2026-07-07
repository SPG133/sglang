import unittest
from types import SimpleNamespace

from sglang.srt.managers.mlfq import (
    MLFQConfig,
    assign_initial_mlfq_level,
    mlfq_sort_key,
    promote_starved_requests,
    record_mlfq_dequeue,
    record_mlfq_enqueue,
    update_mlfq_after_service,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-b-test-cpu")


def make_req(rid, *, level=0, enter=0.0, wait_entry=0.0):
    return SimpleNamespace(
        rid=rid,
        mlfq_level=level,
        mlfq_tokens_in_level=0,
        mlfq_queue_enter_time=enter,
        mlfq_is_queued=True,
        mlfq_last_wait_duration=0.0,
        mlfq_classified_for_queue=False,
        time_stats=SimpleNamespace(wait_queue_entry_time=wait_entry),
    )


class TestMLFQConfig(unittest.TestCase):
    def test_default_config(self):
        config = MLFQConfig()
        self.assertEqual(config.quanta, (8, 16, 32))
        self.assertEqual(config.prefill_thresholds, (64, 512))
        self.assertEqual(config.decode_thresholds, (64, 256))
        self.assertEqual(config.starvation_seconds, 0.2)
        self.assertEqual(config.elastic_long_request_tokens, 32)

    def test_parse_cli_values(self):
        config = MLFQConfig.from_cli_values(
            quanta="2,4,8",
            prefill_thresholds="64,512",
            starvation_seconds=3.5,
        )
        self.assertEqual(config.quanta, (2, 4, 8))
        self.assertEqual(config.prefill_thresholds, (64, 512))
        self.assertEqual(config.starvation_seconds, 3.5)

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            MLFQConfig.from_cli_values(
                quanta="1,0",
                prefill_thresholds="32",
                starvation_seconds=1.0,
            )
        with self.assertRaises(ValueError):
            MLFQConfig.from_cli_values(
                quanta="1,2,4",
                prefill_thresholds="256,32",
                starvation_seconds=1.0,
            )
        with self.assertRaises(ValueError):
            MLFQConfig.from_cli_values(
                quanta="1,2,4",
                prefill_thresholds="32,256",
                starvation_seconds=0,
            )


class TestMLFQAdmissionState(unittest.TestCase):
    def test_initial_classification_uses_work_thresholds(self):
        config = MLFQConfig()
        req = make_req("r")

        assign_initial_mlfq_level(req, 8, config)
        self.assertEqual(req.mlfq_level, 0)
        assign_initial_mlfq_level(req, 128, config)
        self.assertEqual(req.mlfq_level, 1)
        assign_initial_mlfq_level(req, 1024, config)
        self.assertEqual(req.mlfq_level, 2)

    def test_sort_key_level_fifo_rid(self):
        config = MLFQConfig()
        reqs = [
            make_req("c", level=1, enter=1.0),
            make_req("b", level=0, enter=2.0),
            make_req("a", level=0, enter=1.0),
        ]
        reqs.sort(key=lambda r: mlfq_sort_key(r, config))
        self.assertEqual([r.rid for r in reqs], ["a", "b", "c"])

    def test_starvation_uses_current_queue_residence_only(self):
        config = MLFQConfig(starvation_seconds=1.0)
        req = make_req("r", level=2, enter=10.0)

        promote_starved_requests([req], config, timestamp=10.5)
        self.assertEqual(req.mlfq_level, 2)

        promote_starved_requests([req], config, timestamp=11.0)
        self.assertEqual(req.mlfq_level, 0)
        self.assertEqual(req.mlfq_tokens_in_level, 0)
        self.assertEqual(req.mlfq_queue_enter_time, 11.0)

    def test_reenqueue_resets_queue_residence_and_preserves_level(self):
        req = make_req("r", level=2, enter=1.0)
        record_mlfq_dequeue(req, timestamp=2.0)
        self.assertFalse(req.mlfq_is_queued)
        self.assertEqual(req.mlfq_last_wait_duration, 1.0)

        record_mlfq_enqueue(req, timestamp=5.0)
        self.assertTrue(req.mlfq_is_queued)
        self.assertEqual(req.mlfq_level, 2)
        self.assertEqual(req.mlfq_queue_enter_time, 5.0)
        self.assertFalse(req.mlfq_classified_for_queue)

    def test_service_demotes_by_configured_quantum(self):
        config = MLFQConfig(quanta=(2, 4, 8), prefill_thresholds=(32, 256))
        req = make_req("r", level=0)
        update_mlfq_after_service(req, 1, config)
        self.assertEqual((req.mlfq_level, req.mlfq_tokens_in_level), (0, 1))
        update_mlfq_after_service(req, 1, config)
        self.assertEqual((req.mlfq_level, req.mlfq_tokens_in_level), (1, 0))


if __name__ == "__main__":
    unittest.main()
