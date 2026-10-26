"""
main.py: entry point for the TrafficAI API server.
Run from the TrafficAI/ root directory:
    python backend/main.py
"""
import os
import sys

# Ensure the TrafficAI root is on the path (so sim.py etc. are importable)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import uvicorn

if __name__ == "__main__":
    # Change to root dir so relative paths in server.py resolve correctly
    os.chdir(ROOT_DIR)
    # Hosting platforms assign a port at runtime; local use keeps port 8000.
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "backend.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
