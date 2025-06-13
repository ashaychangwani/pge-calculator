#!/usr/bin/env python3
"""
Runner script for PG&E Rate Calculator

This script provides an easy way to run the Streamlit application.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the Streamlit app"""
    try:
        # Get the project root directory
        project_root = Path(__file__).parent
        main_file = project_root / "main.py"
        
        if not main_file.exists():
            print("Error: main.py not found in project directory")
            sys.exit(1)
        
        # Run the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(main_file),
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\nApp stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Error running app: {e}")
        print("Make sure streamlit is installed: pip install streamlit")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main() 