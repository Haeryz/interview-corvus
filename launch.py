#!/usr/bin/env python
"""
Launch script for Interview Corvus
This script ensures the application runs with the correct Poetry environment.
"""
import os
import subprocess
import sys

def main():
    """Launch the Interview Corvus application using Poetry."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Define the command to run
    cmd = ["poetry", "run", "python", "-m", "interview_corvus.main"]
    
    # Print startup message
    print("Starting Interview Corvus...")
    print(f"Working directory: {script_dir}")
    print(f"Command: {' '.join(cmd)}")
    
    # Execute the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Interview Corvus: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
        sys.exit(0)

if __name__ == "__main__":
    main() 