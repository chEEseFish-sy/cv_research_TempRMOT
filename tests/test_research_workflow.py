from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "tools" / "research_workflow" / "plan_gate.py"
HANDOFF = ROOT / "tools" / "research_workflow" / "handoff.py"


def run(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *map(str, args)],
        text=True,
        capture_output=True,
        check=False,
    )


class PlanGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.plan_dir = Path(self.temp.name) / "plan"
        self.plan_dir.mkdir()
        (self.plan_dir / "task_plan.md").write_text(
            "# Plan\n\n### Phase 1\n\nStatus: pending\n", encoding="utf-8"
        )
        self.write_review("GO")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def approve(self) -> subprocess.CompletedProcess[str]:
        return run(
            GATE,
            "approve",
            "--plan-dir",
            self.plan_dir,
            "--scope",
            "Phase 1",
            "--resource",
            "gpu_hours=8",
            "--approved-by",
            "user",
        )

    def write_review(self, decision: str, plan_hash: str | None = None) -> None:
        plan = self.plan_dir / "task_plan.md"
        digest = plan_hash or hashlib.sha256(plan.read_bytes()).hexdigest()
        (self.plan_dir / "feasibility_review.md").write_text(
            f"Decision: {decision}\nPlan-SHA256: `{digest}`\n", encoding="utf-8"
        )

    def test_approve_and_verify(self) -> None:
        self.assertEqual(self.approve().returncode, 0)
        result = run(
            GATE,
            "verify",
            "--plan-dir",
            self.plan_dir,
            "--phase",
            "Phase 1",
            "--resource",
            "gpu_hours=4",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["ok"])

    def test_missing_approval_fails(self) -> None:
        result = run(GATE, "verify", "--plan-dir", self.plan_dir, "--phase", "Phase 1")
        self.assertEqual(result.returncode, 2)

    def test_changed_plan_fails(self) -> None:
        self.assertEqual(self.approve().returncode, 0)
        (self.plan_dir / "task_plan.md").write_text("# Changed\n", encoding="utf-8")
        result = run(
            GATE,
            "verify",
            "--plan-dir",
            self.plan_dir,
            "--phase",
            "Phase 1",
            "--resource",
            "gpu_hours=4",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("hash mismatch", result.stderr)

    def test_scope_and_resource_failures(self) -> None:
        self.assertEqual(self.approve().returncode, 0)
        wrong_phase = run(
            GATE,
            "verify",
            "--plan-dir",
            self.plan_dir,
            "--phase",
            "Phase 2",
            "--resource",
            "gpu_hours=4",
        )
        self.assertEqual(wrong_phase.returncode, 2)
        exceeded = run(
            GATE,
            "verify",
            "--plan-dir",
            self.plan_dir,
            "--phase",
            "Phase 1",
            "--resource",
            "gpu_hours=9",
        )
        self.assertEqual(exceeded.returncode, 2)

    def test_approval_requires_go_review_declared_scope_and_resources(self) -> None:
        self.write_review("REVISE")
        revise = self.approve()
        self.assertEqual(revise.returncode, 2)
        self.assertIn("not GO", revise.stderr)

        self.write_review("GO")
        undeclared = run(
            GATE,
            "approve",
            "--plan-dir",
            self.plan_dir,
            "--scope",
            "Phase 9",
            "--resource",
            "gpu_hours=4",
            "--approved-by",
            "user",
        )
        self.assertEqual(undeclared.returncode, 2)
        no_resource = run(
            GATE,
            "approve",
            "--plan-dir",
            self.plan_dir,
            "--scope",
            "Phase 1",
            "--approved-by",
            "user",
        )
        self.assertEqual(no_resource.returncode, 2)

    def test_nonfinite_resources_and_stale_review_fail(self) -> None:
        nonfinite = run(
            GATE,
            "approve",
            "--plan-dir",
            self.plan_dir,
            "--scope",
            "Phase 1",
            "--resource",
            "gpu_hours=NaN",
            "--approved-by",
            "user",
        )
        self.assertEqual(nonfinite.returncode, 2)
        self.assertIn("finite", nonfinite.stderr)

        self.write_review("GO", "0" * 64)
        stale = self.approve()
        self.assertEqual(stale.returncode, 2)
        self.assertIn("review is stale", stale.stderr)


class HandoffTests(unittest.TestCase):
    def test_thresholds(self) -> None:
        below = run(HANDOFF, "check", "--context-percent", "74", "--tool-calls", "39")
        self.assertFalse(json.loads(below.stdout)["handoff_required"])
        threshold = run(HANDOFF, "check", "--substantive-turns", "12")
        self.assertTrue(json.loads(threshold.stdout)["handoff_required"])
        papers = run(HANDOFF, "check", "--l3-papers", "3")
        self.assertTrue(json.loads(papers.stdout)["handoff_required"])

    def test_create_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "payload.json"
            payload.write_text(
                json.dumps(
                    {
                        "agent": "research_dialogue",
                        "objective": "Evaluate order sensitivity",
                        "current_state": "Evidence has been gathered.",
                        "next_action": "Compare swap effect with paraphrase noise.",
                        "decisions": ["Use the two-sample pilot first."],
                        "evidence": ["docs/research/example.md"],
                    }
                ),
                encoding="utf-8",
            )
            result = run(
                HANDOFF,
                "create",
                "--payload",
                payload,
                "--root",
                root / "handoffs",
                "--apply",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            handoff = Path(data["handoff"])
            resume = Path(data["resume_prompt"])
            self.assertTrue(handoff.is_file())
            self.assertTrue(resume.is_file())
            self.assertIn("Exact next action", handoff.read_text(encoding="utf-8"))
            self.assertIn(str(handoff), resume.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
