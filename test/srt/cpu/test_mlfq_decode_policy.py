import unittest
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.utils import MetadataBuffers
from sglang.srt.managers.mlfq import (
    DecodeMLFQStats,
    MLFQConfig,
    assign_initial_decode_mlfq_level,
    maybe_elastic_promote,
    mlfq_sort_key,
    update_decode_mlfq_after_service,
)
from sglang.srt.server_args import ServerArgs


class FakeScheduler:
    def __init__(self, mlfq_config):
        self.mlfq_config = mlfq_config

    def is_decode_mlfq_enabled(self):
        return True

    def _prepare_decode_mlfq_queue(self, reqs):
        for req in reqs:
            if not getattr(req, "mlfq_decode_initialized", False):
                assign_initial_decode_mlfq_level(req, self.mlfq_config)


def make_req(
    rid,
    *,
    max_new_tokens=128,
    output_len=0,
    level=2,
    queue_enter_time=0.0,
):
    return SimpleNamespace(
        rid=rid,
        sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
        output_ids=[0] * output_len,
        mlfq_level=level,
        mlfq_tokens_in_level=0,
        mlfq_queue_enter_time=queue_enter_time,
        mlfq_is_queued=True,
        mlfq_classified_for_queue=False,
        mlfq_decode_initialized=False,
        mlfq_elastic_promoted=False,
        predicted_slowdown_at_last_queue_check=0.0,
        elastic_effective_threshold=0.0,
        scheduler_enqueue_time=queue_enter_time,
        decode_batch_wall_time_attributed=0.0,
        decode_execution_time=0.0,
        time_stats=SimpleNamespace(
            wait_queue_entry_time=0.0,
            decode_prealloc_queue_entry_time=queue_enter_time,
            decode_transfer_queue_entry_time=0.0,
        ),
    )


class TestMLFQDecodePolicy(unittest.TestCase):
    def test_prefill_requested_mlfq_effective_fcfs(self):
        args = ServerArgs.__new__(ServerArgs)
        args.schedule_policy = "mlfq"
        args.disaggregation_mode = "prefill"

        ServerArgs._handle_effective_schedule_policy(args)

        self.assertEqual(args.requested_schedule_policy, "mlfq")
        self.assertEqual(args.effective_schedule_policy, "fcfs")

    def test_null_requested_mlfq_effective_fcfs(self):
        args = ServerArgs.__new__(ServerArgs)
        args.schedule_policy = "mlfq"
        args.disaggregation_mode = "null"

        ServerArgs._handle_effective_schedule_policy(args)

        self.assertEqual(args.requested_schedule_policy, "mlfq")
        self.assertEqual(args.effective_schedule_policy, "fcfs")

    def test_decode_requested_mlfq_stays_effective_mlfq(self):
        args = ServerArgs.__new__(ServerArgs)
        args.schedule_policy = "mlfq"
        args.disaggregation_mode = "decode"

        ServerArgs._handle_effective_schedule_policy(args)

        self.assertEqual(args.requested_schedule_policy, "mlfq")
        self.assertEqual(args.effective_schedule_policy, "mlfq")

    def test_ordinary_waiting_queue_sorts_by_mlfq_key(self):
        config = MLFQConfig()
        reqs = [
            make_req("low", level=2, queue_enter_time=1.0),
            make_req("high", level=0, queue_enter_time=2.0),
            make_req("mid", level=1, queue_enter_time=0.5),
        ]

        reqs.sort(key=lambda req: mlfq_sort_key(req, config))

        self.assertEqual([req.rid for req in reqs], ["high", "mid", "low"])

    def test_prealloc_and_retracted_queues_sort_after_local_decode_init(self):
        config = MLFQConfig.from_cli_values(
            quanta="1,2,4",
            prefill_thresholds="32,256",
            decode_thresholds="32,256",
            starvation_seconds=1.0,
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.scheduler = FakeScheduler(config)
        long_req = make_req("long", max_new_tokens=300, queue_enter_time=0.0)
        short_req = make_req("short", max_new_tokens=16, queue_enter_time=1.0)
        mid_req = make_req("mid", max_new_tokens=128, queue_enter_time=2.0)
        queue.queue = [SimpleNamespace(req=long_req), SimpleNamespace(req=short_req)]
        queue.retracted_queue = [long_req, mid_req, short_req]

        DecodePreallocQueue._sort_decode_queues_for_mlfq(queue)

        self.assertEqual([entry.req.rid for entry in queue.queue], ["short", "long"])
        self.assertEqual(
            [req.rid for req in queue.retracted_queue], ["short", "mid", "long"]
        )
        self.assertTrue(short_req.mlfq_decode_initialized)
        self.assertTrue(long_req.mlfq_decode_initialized)

    def test_prefill_metadata_marks_mlfq_state_invalid(self):
        buffers = MetadataBuffers(
            size=1,
            hidden_size=1,
            hidden_states_dtype=torch.float32,
        )
        req = SimpleNamespace(
            metadata_buffer_index=0,
            output_ids=[7],
            cached_tokens=0,
            cached_tokens_device=0,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            return_logprob=False,
            hidden_states_tensor=None,
            bootstrap_room=123,
            time_stats=SimpleNamespace(
                prefill_bootstrap_queue_entry_time=1.0,
                wait_queue_entry_time=2.0,
                forward_entry_time=3.0,
                prefill_finished_time=4.0,
                prefill_transfer_queue_entry_time=5.0,
                prefill_kv_transfer_finish_time=0.0,
            ),
            scheduler_enqueue_time=0.5,
            prefill_batch_wall_time_attributed=0.1,
            kv_transfer_time=0.0,
            attributed_batch_wall_time=0.1,
            prefill_execution_time=0.1,
            actual_execution_time=0.1,
            mlfq_classified_for_queue=True,
            mlfq_level=0,
            mlfq_tokens_in_level=9,
            origin_input_ids=[1, 2, 3],
        )

        buffers.set_buf(req)

        values = buffers.prefill_timing_info[0].tolist()
        fields = buffers.prefill_timing_fields
        self.assertEqual(values[fields.index("prefill_mlfq_state_valid")], 0.0)
        self.assertEqual(values[fields.index("prefill_mlfq_level")], 0.0)
        self.assertEqual(values[fields.index("prefill_mlfq_tokens_in_level")], 0.0)

    def test_decode_local_init_ignores_prefill_mlfq_state(self):
        config = MLFQConfig()
        req = make_req("decode-local", max_new_tokens=300, level=0)
        req.pd_prefill_timing_info = {
            "prefill_mlfq_state_valid": 1,
            "prefill_mlfq_level": 0,
            "prefill_mlfq_tokens_in_level": 99,
        }

        assign_initial_decode_mlfq_level(req, config)

        self.assertEqual(req.mlfq_level, 2)
        self.assertEqual(req.mlfq_tokens_in_level, 0)

    def test_missing_metadata_uses_same_decode_local_init(self):
        config = MLFQConfig()
        fake_req = make_req("fake", max_new_tokens=16, level=2)
        missing_req = make_req("missing", max_new_tokens=16, level=2)

        assign_initial_decode_mlfq_level(fake_req, config)
        assign_initial_decode_mlfq_level(missing_req, config)

        self.assertEqual(fake_req.mlfq_level, missing_req.mlfq_level)
        self.assertEqual(fake_req.mlfq_level, 0)

    def test_decode_update_keeps_short_remaining_work_high_priority(self):
        config = MLFQConfig(quanta=(1, 2, 4), prefill_thresholds=(64, 512))
        req = make_req("short", max_new_tokens=3, output_len=1, level=0)

        update_decode_mlfq_after_service(req, 1, config)

        self.assertEqual(req.mlfq_level, 0)
        self.assertEqual(req.mlfq_tokens_in_level, 0)

    def test_decode_update_still_demotes_long_remaining_work(self):
        config = MLFQConfig(quanta=(1, 2, 4), prefill_thresholds=(64, 512))
        req = make_req("long", max_new_tokens=512, output_len=1, level=0)

        update_decode_mlfq_after_service(req, 1, config)

        self.assertEqual(req.mlfq_level, 1)
        self.assertEqual(req.mlfq_tokens_in_level, 0)

    def test_completed_slowdown_mean(self):
        stats = DecodeMLFQStats()

        stats.record_completed_request(
            d_flow_time=1.0, d_service_time=1.0, output_tokens=1, epsilon=1e-6
        )
        stats.record_completed_request(
            d_flow_time=3.0, d_service_time=1.0, output_tokens=1, epsilon=1e-6
        )

        self.assertEqual(stats.completed_count, 2)
        self.assertEqual(stats.completed_slowdown_mean, 2.0)

    def test_elastic_promotes_long_request_when_predicted_slowdown_crosses_threshold(self):
        config = MLFQConfig(
            elastic_slowdown_multiplier=1.2,
            elastic_long_request_tokens=10,
            elastic_min_completed_requests=0,
        )
        stats = DecodeMLFQStats(
            completed_count=1,
            completed_slowdown_sum=1.0,
            decode_seconds_per_token_ewma=1.0,
        )
        req = make_req("long", max_new_tokens=100, level=2)

        promoted = maybe_elastic_promote(
            req,
            now=30.0,
            d_queue_entry_time=0.0,
            stats=stats,
            config=config,
        )

        self.assertTrue(promoted)
        self.assertEqual(req.mlfq_level, 0)
        self.assertEqual(req.mlfq_tokens_in_level, 0)
        self.assertTrue(req.mlfq_elastic_promoted)
        self.assertAlmostEqual(req.predicted_slowdown_at_last_queue_check, 1.3)
        self.assertAlmostEqual(req.elastic_effective_threshold, 1.2)

    def test_elastic_does_not_promote_below_threshold_short_or_under_sampled(self):
        config = MLFQConfig(
            elastic_slowdown_multiplier=1.2,
            elastic_long_request_tokens=10,
            elastic_min_completed_requests=1,
        )
        stats = DecodeMLFQStats(
            completed_count=1,
            completed_slowdown_sum=1.0,
            decode_seconds_per_token_ewma=1.0,
        )
        below = make_req("below", max_new_tokens=100, level=2)
        short = make_req("short", max_new_tokens=5, level=2)
        under_sampled = make_req("under", max_new_tokens=100, level=2)

        self.assertFalse(
            maybe_elastic_promote(
                below,
                now=19.0,
                d_queue_entry_time=0.0,
                stats=stats,
                config=config,
            )
        )
        self.assertFalse(
            maybe_elastic_promote(
                short,
                now=30.0,
                d_queue_entry_time=0.0,
                stats=stats,
                config=config,
            )
        )
        under_config = MLFQConfig(
            elastic_slowdown_multiplier=1.2,
            elastic_long_request_tokens=10,
            elastic_min_completed_requests=2,
        )
        self.assertFalse(
            maybe_elastic_promote(
                under_sampled,
                now=30.0,
                d_queue_entry_time=0.0,
                stats=stats,
                config=under_config,
            )
        )

    def test_elastic_promotion_is_not_repeated_for_same_request(self):
        config = MLFQConfig(
            elastic_slowdown_multiplier=1.2,
            elastic_long_request_tokens=10,
            elastic_min_completed_requests=0,
        )
        stats = DecodeMLFQStats(
            completed_count=1,
            completed_slowdown_sum=1.0,
            decode_seconds_per_token_ewma=1.0,
        )
        req = make_req("once", max_new_tokens=100, level=2)

        first = maybe_elastic_promote(
            req,
            now=30.0,
            d_queue_entry_time=0.0,
            stats=stats,
            config=config,
        )
        second = maybe_elastic_promote(
            req,
            now=40.0,
            d_queue_entry_time=0.0,
            stats=stats,
            config=config,
        )

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(req.mlfq_level, 0)


if __name__ == "__main__":
    unittest.main()
