"""Exercise the shell launcher with fake GPUs and a fake Python executable."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ContactLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        source = Path(__file__).resolve().parents[1] / "scripts/0_interactvlm.sh"
        self.launcher = scripts / source.name
        shutil.copyfile(source, self.launcher)
        cp = self.root / "prep/Contact-Predictor"
        (cp / "data/hcontact_vitruvian").mkdir(parents=True)
        (cp / "data/smpl_to_smplx.pkl").touch()
        self.model = cp / "trained_models/test"
        self.model.mkdir(parents=True)
        (self.model / "config.json").write_text("{}")
        for sequence in ["a", "b"]:
            (self.root / "input_img" / sequence).mkdir(parents=True)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.executable("nvidia-smi", '#!/bin/bash\nprintf "0\\n2\\n"\n')
        python = self.executable(
            "fake-python",
            '#!/bin/bash\nprintf "%s|%s|%s\\n" "$0" "$CUDA_VISIBLE_DEVICES" "$*" '
            '>> "$CONTACT_TEST_LOG"\nexit "${CONTACT_TEST_EXIT:-0}"\n',
        )
        venv_bin = self.root / ".venv-contact/bin"
        venv_bin.mkdir(parents=True)
        self.python = venv_bin / "python"
        self.python.symlink_to(python)
        self.log = self.root / "calls.txt"

    def executable(self, name, text):
        path = self.bin / name
        path.write_text(text)
        path.chmod(0o755)
        return path

    def run_launcher(self, **overrides):
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.update(
            PATH=f"{self.bin}:{os.environ['PATH']}",
            CONTACT_PYTHON=str(self.python), CONTACT_MODEL_PATH=str(self.model),
            CONTACT_QUANTIZATION="4bit", CONTACT_TEST_LOG=str(self.log),
        )
        env.update(overrides)
        return subprocess.run(
            ["bash", str(self.launcher), str(self.root / "input"), "box"],
            env=env, capture_output=True, text=True, timeout=10,
        )

    def test_venv_noncontiguous_gpus_and_quantization(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [line.split("|", 2) for line in self.log.read_text().splitlines()]
        self.assertEqual({call[1] for call in calls}, {"0", "2"})
        self.assertEqual(len(calls), 2)
        for executable, _, arguments in calls:
            self.assertEqual(executable, str(self.python))
            self.assertIn("--obj box", arguments)
            self.assertIn("--precision fp16 --load_in_4bit", arguments)

    def test_worker_failure_is_propagated(self):
        result = self.run_launcher(CONTACT_TEST_EXIT="7")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Contact inference failed", result.stderr)
        self.assertNotIn("All jobs finished", result.stdout)

    def test_missing_model_does_not_launch_inference(self):
        (self.model / "config.json").unlink()
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Contact model missing", result.stderr)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
