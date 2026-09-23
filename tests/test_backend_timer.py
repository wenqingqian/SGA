# Copyright (c) 2026, SGA framework contributors.
"""backend_timer unit tests.

CPU-only: the CUDA guards in begin/end degrade to no-ops without a
device, so the record path is exercised on CPU.
"""

import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sga import backend_timer  # noqa: E402


class _ResetMixin:
    def setUp(self):
        backend_timer.reset()

    tearDown = setUp


class TestOffState(_ResetMixin, unittest.TestCase):
    def test_off_defaults(self):
        self.assertFalse(backend_timer.active())
        self.assertIsNone(backend_timer.begin())

    def test_end_with_none_token_is_noop(self):
        backend_timer.end(None, "sparse", k=1)

    def test_install_fullsync_wrap_noop_when_off(self):
        class Chunk:
            def finish_grad_sync(self, force_all_reduce=False):
                return "orig"

        backend_timer._wrap_finish_grad_sync(Chunk)
        self.assertEqual(Chunk().finish_grad_sync(), "orig")

    def test_set_iter_noop_when_off(self):
        backend_timer.set_iter(7)


class TestRecords(_ResetMixin, unittest.TestCase):
    def test_configure_writes_rank_file(self):
        with tempfile.TemporaryDirectory() as d:
            backend_timer.configure(d, rank=3)
            self.assertTrue(backend_timer.active())
            name = os.path.join(d, "backend_timer_rank3.jsonl")
            self.assertTrue(os.path.exists(name))

            backend_timer.set_iter(12)
            tok = backend_timer.begin()
            time.sleep(0.002)
            backend_timer.end(tok, "sparse", backend="b", k=5)
            backend_timer.end(None, "prescale", k=5)

            with open(name) as f:
                lines = [json.loads(x) for x in f if x.strip()]
            self.assertEqual(len(lines), 1)
            rec = lines[0]
            self.assertEqual(rec["phase"], "sparse")
            self.assertEqual(rec["rank"], 3)
            self.assertEqual(rec["iter"], 12)
            self.assertEqual(rec["seq"], 0)
            self.assertEqual(rec["backend"], "b")
            self.assertEqual(rec["k"], 5)
            self.assertGreater(rec["wall_ms"], 0.0)
            self.assertIn("ts", rec)

    def test_double_configure_refused(self):
        with tempfile.TemporaryDirectory() as d:
            backend_timer.configure(d, rank=0)
            with self.assertRaises(RuntimeError):
                backend_timer.configure(d, rank=0)

    def test_empty_path_refused(self):
        with self.assertRaises(ValueError):
            backend_timer.configure("  ", rank=0)

    def test_seq_monotonic(self):
        with tempfile.TemporaryDirectory() as d:
            backend_timer.configure(d, rank=0)
            for _ in range(3):
                backend_timer.end(backend_timer.begin(), "fullsync")
            with open(os.path.join(d, "backend_timer_rank0.jsonl")) as f:
                seqs = [json.loads(x)["seq"] for x in f if x.strip()]
            self.assertEqual(seqs, [0, 1, 2])


class _FakeDDP:
    """The finish_grad_sync surface of megatron's DDP (just enough)."""

    def __init__(self, overlap=False):
        class Cfg:
            pass

        self.ddp_config = Cfg()
        self.ddp_config.overlap_grad_reduce = overlap
        self.calls = 0

    def finish_grad_sync(self, force_all_reduce=False):
        self.calls += 1
        return "done"


class TestFullsyncWrap(_ResetMixin, unittest.TestCase):
    def _wrap(self):
        # a fresh subclass per test: the wrap is idempotent per class, so
        # sharing one class across tests would leak the wrapper
        cls = type("Chunk", (_FakeDDP,), {})
        backend_timer._wrap_finish_grad_sync(cls)
        return cls

    def test_wrap_records_fullsync_phase(self):
        with tempfile.TemporaryDirectory() as d:
            backend_timer.configure(d, rank=0)
            backend_timer.set_iter(8)
            cls = self._wrap()
            chunk = cls()
            self.assertEqual(chunk.finish_grad_sync(force_all_reduce=True),
                             "done")
            self.assertEqual(chunk.calls, 1)
            with open(os.path.join(d, "backend_timer_rank0.jsonl")) as f:
                rec = json.loads(f.read())
            self.assertEqual(rec["phase"], "fullsync")
            self.assertEqual(rec["iter"], 8)
            self.assertEqual(rec["backend"], "megatron_ddp")
            self.assertNotIn("error", rec)

    def test_idempotent_single_wrap(self):
        with tempfile.TemporaryDirectory() as d:
            backend_timer.configure(d, rank=0)
            cls = self._wrap()
            backend_timer._wrap_finish_grad_sync(cls)
            chunk = cls()
            chunk.finish_grad_sync()
            with open(os.path.join(d, "backend_timer_rank0.jsonl")) as f:
                self.assertEqual(len([x for x in f if x.strip()]), 1)

    def test_overlap_refused_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            backend_timer.configure(d, rank=0)
            cls = self._wrap()
            chunk = cls(overlap=True)
            with self.assertRaises(RuntimeError):
                chunk.finish_grad_sync()
            self.assertEqual(chunk.calls, 0)

    def test_orig_raise_propagates_and_still_records_error(self):
        with tempfile.TemporaryDirectory() as d:
            backend_timer.configure(d, rank=0)

            class Raising(_FakeDDP):
                def finish_grad_sync(self, force_all_reduce=False):
                    self.calls += 1
                    raise RuntimeError("orig sync failed")

            cls = type("Chunk", (Raising,), {})
            backend_timer._wrap_finish_grad_sync(cls)
            chunk = cls()
            with self.assertRaises(RuntimeError):
                chunk.finish_grad_sync()
            self.assertEqual(chunk.calls, 1)
            with open(os.path.join(d, "backend_timer_rank0.jsonl")) as f:
                rec = json.loads(f.read())
            self.assertEqual(rec["phase"], "fullsync")
            self.assertIs(rec["error"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
