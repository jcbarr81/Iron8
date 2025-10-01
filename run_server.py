import os, sys
# Ensure 'src' is importable when running `uvicorn run_server:app`
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from baseball_sim.serve.api import app
