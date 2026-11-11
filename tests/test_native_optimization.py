import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from native_backbone_metrics import select_horizon, summarize
from run_native_optimization import Device, conditions, run_groups, study_plan
from run_native_osc import BACKBONE_PROTOCOL, verify_gpu_placement


class NativeOptimizationTests(unittest.TestCase):
    def test_plan_partitions_and_sizes(self):
        plan = study_plan()
        phases = {phase: conditions(plan, phase) for phase in ("pilot", "development", "confirmation")}
        self.assertEqual([sum(len(job.variants) for job in jobs) for jobs in phases.values()], [18, 54, 72])
        self.assertEqual(sum(len(job.variants) for job in conditions(plan, "confirmation", 25)), 108)
        starts = [(job.case["scene"], job.case["init"]) for jobs in phases.values() for job in jobs]
        self.assertEqual(len(starts), len(set(starts)))

    def test_selection_respects_scene_regressions_and_call_tiebreaker(self):
        groups = [
            {"case": {"scene": scene}, "episodes": {
                "pi05-10": {"success": scene == 8, "vla_calls": 10},
                "pi05-5": {"success": True, "vla_calls": 20},
                "pi05-25": {"success": True, "vla_calls": 4},
            }} for scene in (1, 2, 8)
        ]
        self.assertEqual(select_horizon(groups)["selected_chunk"], 25)
        groups[-1]["episodes"]["pi05-25"]["success"] = False
        self.assertEqual(select_horizon(groups)["selected_chunk"], 5)
        groups[0]["episodes"]["pi05-5"]["success"] = False
        self.assertEqual(select_horizon(groups)["selected_chunk"], 10)

    def test_gpu_placement_checks_both_live_processes(self):
        pid = os.getpid()
        example = f"""<nvidia_smi_log><driver_version>test</driver_version>
        <gpu><uuid>GPU-1</uuid><processes>
        <process_info><pid>{pid}</pid><type>C</type></process_info>
        <process_info><pid>123</pid><type>G</type></process_info>
        </processes></gpu></nvidia_smi_log>"""
        with (patch("run_native_osc.subprocess.check_output", return_value=example.encode()),
              patch.dict(os.environ, {"MUJOCO_EGL_DEVICE_ID": "0"})):
            self.assertEqual(verify_gpu_placement("GPU-1", 123)["uuid"], "GPU-1")
            with self.assertRaises(RuntimeError):
                verify_gpu_placement("GPU-1", 124)
            with self.assertRaises(RuntimeError):
                verify_gpu_placement("GPU-2", 123)

    def test_resume_preserves_gpu_pairing_and_excludes_infrastructure_failure(self):
        plan = study_plan()
        jobs = conditions(plan, "pilot")[:2]
        devices = [Device("GPU-A", 0), Device("GPU-B", 1)]
        failed = False

        def fake_episode(command, **kwargs):
            nonlocal failed

            def value(name):
                return command[command.index(f"--{name}") + 1]

            output = Path(value("output"))
            policy = value("policy")
            scene, seed, init = (int(value(name)) for name in ("scene", "seed", "init-index"))
            if policy == "pi05" and scene == 1 and not failed:
                failed = True
                return subprocess.CompletedProcess(command, 1)
            output.mkdir()
            gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            success = policy == "pi05"
            outcome = "simulator_success" if success else "simulator_done"
            state_hash = sha256(f"{scene}-{init}".encode()).hexdigest()
            episode = {
                "project_commit": plan["project_commit"], "step_budget": 520, "chunk_horizon": 10,
                "method": "direct_vla", "method_protocol": BACKBONE_PROTOCOL,
                "task_id": f"libero/libero-10-{scene:02d}", "seed": seed, "init_state_index": init,
                "controller": "OSC_POSE", "control_hz": 20, "settle_frames": 10,
                "success": success, "terminal_reason": outcome, "steps": 1,
                "vla_calls": 1, "visual_calls": 0, "episode_elapsed_s": 1,
                "policy_inference_s": 0.5, "cuda_peak_reserved_mib": 100,
                "instruction": "full official instruction", "init_state_sha256": state_hash,
                "bddl_sha256": "a" * 64, "sim_state_sha256": "b" * 64,
                "observation_sha256": {"camera1": "c" * 64, "camera2": "d" * 64},
                "gpu_placement": {"uuid": gpu},
            }
            if policy == "pi05":
                episode.update(model=plan["pi05"]["repository"], model_artifacts=[plan["pi05"]])
            else:
                episode["model"] = f"{plan['smolvla']['repository']}@{plan['smolvla']['revision']}"
            (output / "episode.json").write_text(json.dumps(episode), encoding="utf-8")
            trace = {"actions": [[0] * 7], "start_step": 0, "executed": 1, "success": success,
                     "done": not success, "instruction": episode["instruction"], "audit_goals": []}
            (output / "actions.jsonl").write_text(json.dumps(trace) + "\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(output=root, workspace=root, roboenv=root, sim_python=Path(sys.executable))
            (root / "pilot" / devices[1].uuid / jobs[0].name).mkdir(parents=True)
            with patch("compare_native_backbones.subprocess.run", side_effect=fake_episode) as launch:
                groups = run_groups(args, plan, jobs, devices, None)["pilot"]
                self.assertEqual(launch.call_count, 5)
                self.assertEqual(groups[0]["gpu_uuid"], devices[1].uuid)
                self.assertEqual(groups[0]["infrastructure_failed_attempts"], 1)
                self.assertEqual(summarize(groups, "smolvla-10")["comparisons"]["pi05-10"]["net_wins"], 2)
                for group in groups:
                    self.assertEqual({row["gpu_placement"]["uuid"] for row in group["episodes"].values()}, {group["gpu_uuid"]})
                self.assertEqual(run_groups(args, plan, jobs, devices, None)["pilot"], groups)
                self.assertEqual(launch.call_count, 5)
