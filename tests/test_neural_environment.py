import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import numpy as np
from modules.neural_portrait import NeuralPortraitJob


class EnvironmentSelectionTests(unittest.TestCase):
    def check_interpreter(self, available, override, expected):
        process = Mock()
        process.poll.return_value = 0
        process.wait.return_value = 0
        with patch.dict(os.environ, {}, clear=True), patch('modules.neural_portrait.importlib.util.find_spec', return_value=object() if available else None), patch('modules.neural_portrait.Path.is_file', return_value=True), patch('modules.neural_portrait.subprocess.Popen', return_value=process) as popen:
            if override:
                os.environ['AIRBOARD_NEURAL_PYTHON'] = override
            job = NeuralPortraitJob(np.zeros((32,32,3), np.uint8), 'van_gogh', .3)
            self.assertEqual(popen.call_args.args[0][0], expected)
            job.log.close()
            job.temp.cleanup()

    def test_current_environment(self):
        self.check_interpreter(True, None, sys.executable)

    def test_explicit_override(self):
        self.check_interpreter(True, '/custom/python', '/custom/python')

    def test_legacy_environment(self):
        expected = Path(sys.prefix).parent / 'ai-learning' / ('python.exe' if os.name == 'nt' else 'bin/python')
        self.check_interpreter(False, None, str(expected))
