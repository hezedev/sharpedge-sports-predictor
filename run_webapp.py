#!/usr/bin/env python3
"""Launch the SharpEdge web app. Run from the sports_predictor directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from webapp.app import app

if __name__ == "__main__":
    print("\n  🏆 SharpEdge — AI Sports Predictor")
    print("  ─────────────────────────────────")
    print("  Open: http://localhost:5000\n")
    app.run(debug=False, port=5000, threaded=True)
