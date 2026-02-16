#!/usr/bin/env python
"""Launch the ASI Spectral Catalog Dash dashboard."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.app import app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch the ASI Spectral Catalog dashboard"
    )
    parser.add_argument("--port", type=int, default=8050, help="Port to serve on")
    parser.add_argument(
        "--debug", action="store_true", default=True, help="Enable debug mode"
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Run with gunicorn (4 workers) instead of the Dash dev server",
    )
    args = parser.parse_args()

    if args.production:
        import subprocess

        subprocess.run(
            [
                "gunicorn",
                "dashboard.app:server",
                "-b",
                f"0.0.0.0:{args.port}",
                "-w",
                "4",
            ]
        )
    else:
        app.run(debug=args.debug, port=args.port)
