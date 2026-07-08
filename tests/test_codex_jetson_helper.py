import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "scripts" / "codex" / "Invoke-JetsonScript.ps1"
WRAPPER_PATH = REPO_ROOT / "scripts" / "codex" / "invoke-jetson-script.cmd"


class CodexJetsonHelperTests(unittest.TestCase):
    def test_helper_exists(self):
        self.assertTrue(HELPER_PATH.is_file(), f"Missing helper script: {HELPER_PATH}")
        self.assertTrue(WRAPPER_PATH.is_file(), f"Missing wrapper script: {WRAPPER_PATH}")

    def test_helper_normalizes_line_endings_and_supports_conda(self):
        content = HELPER_PATH.read_text(encoding="utf-8")

        self.assertIn("-replace \"`r`n\", \"`n\"", content)
        self.assertIn("-replace \"`r\", \"`n\"", content)
        self.assertIn("System.Text.UTF8Encoding -ArgumentList $false", content)
        self.assertIn("[System.IO.File]::WriteAllText($localTemp, $content, $utf8NoBom)", content)
        self.assertIn("source '$CondaRoot/etc/profile.d/conda.sh'", content)
        self.assertIn("conda activate '$CondaEnv'", content)
        self.assertIn("scp $localTemp \"${HostName}:$remoteTemp\"", content)
        self.assertIn("ssh $HostName $remoteCommand", content)

    def test_cmd_wrapper_copies_helper_to_local_temp_before_execution(self):
        content = WRAPPER_PATH.read_text(encoding="utf-8")

        self.assertIn('copy /Y "%HELPER_SRC%" "%HELPER_DST%"', content)
        self.assertIn('powershell -NoProfile -ExecutionPolicy Bypass -File "%HELPER_DST%" %*', content)
        self.assertIn('del "%HELPER_DST%"', content)


if __name__ == "__main__":
    unittest.main()
