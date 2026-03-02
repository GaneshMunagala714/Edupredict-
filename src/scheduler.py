"""
EduPredict Data Scheduler
Runs data fetching and cleaning on a schedule.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from fetcher import check_all_sources
from cleaner import run_all_cleaners


def run_scheduled_job():
    """
    Run the complete data pipeline:
    1. Check all sources for updates
    2. Clean any new data
    3. Log results
    """
    print("\n" + "=" * 60)
    print("EduPredict Scheduled Data Update")
    print("=" * 60)
    
    # Step 1: Fetch
    print("\n[1/3] Checking data sources...")
    new_files = check_all_sources()
    
    # Step 2: Clean (only if new data found)
    if new_files:
        print(f"\n[2/3] Cleaning {len(new_files)} new file(s)...")
        cleaned = run_all_cleaners()
        print(f"  Cleaned {len(cleaned)} file(s)")
    else:
        print("\n[2/3] No new data to clean")
    
    # Step 3: Summary
    print("\n[3/3] Update complete")
    print("=" * 60)
    
    return len(new_files) > 0


def setup_cron_job():
    """
    Print instructions for setting up cron job.
    """
    script_path = Path(__file__).resolve()
    
    print("""
To schedule weekly runs, add this to your crontab:

# EduPredict weekly data update (Sundays at 2 AM)
0 2 * * 0 cd {dir} && /usr/bin/python3 src/scheduler.py >> logs/scheduler.log 2>&1

Or run manually:
  python src/scheduler.py
""".format(dir=script_path.parent.parent))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        setup_cron_job()
    else:
        run_scheduled_job()