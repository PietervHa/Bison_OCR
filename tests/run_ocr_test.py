#!/usr/bin/env python
"""
Wrapper script to run paddleocr_test.py with proper environment variables
"""
import subprocess
import sys
import os

# Set environment variables before running the subprocess
env = os.environ.copy()
env['PADDLE_ENABLE_ONEDNN'] = '0'
env['PADDLE_CUDNN_DETERMINISTIC'] = '0'
env['PADDLE_MKL_NUM_THREADS'] = '1'
env['PADDLE_NUM_THREADS'] = '4'
env['PADDLE_DISABLE_FAST_OPERATORS'] = '1'
env['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
env['FLAGS_call_stack_level'] = '2'
env['FLAGS_print_phi_ops_logs'] = '0'

# Run the actual script
result = subprocess.run(
    [sys.executable, 'paddleocr_test_core.py'],
    env=env,
    cwd=os.path.dirname(os.path.abspath(__file__))
)

sys.exit(result.returncode)

