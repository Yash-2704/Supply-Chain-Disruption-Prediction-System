"""Launch the merged risk dashboard (Streamlit).

If the export data is missing, tells you to build it first.
Run:  python scripts/run_risk_dashboard.py
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP = PROJECT_ROOT / "risk_dashboard" / "app.py"
SCORES = PROJECT_ROOT / "dashboard_export" / "processed" / "risk_scores.csv"

if __name__ == "__main__":
    if not SCORES.exists():
        print("⚠️  No dashboard data found. Build it first:")
        print("    python scripts/build_dashboard_data.py\n")
    print("🌐 Launching dashboard at http://localhost:8501 …")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP),
                    "--server.headless", "true"])
