import os
import subprocess
from pathlib import Path

import csv

from dotenv import load_dotenv

load_dotenv()

CSV_FILE = os.getenv('DM_BUOYS_TO_PROC_PATH')
ARCHIVE_BASE_PATH = os.getenv('ARCHIVE_BASE_PATH')
DM_DATA_PATH = os.getenv('DM_DATA_PATH')
RAWDATAFOLDER = "log"
SUBFOLDER = "processed_py"


def main():

    with open(CSV_FILE) as f:
        print("test")
        reader = csv.DictReader(f)
        
        for row in reader:
            
            if row["archive"] != "1":
                continue

            print("\n", row["datapath"])

            #  DEPLOYMENT FOLDER ------------------------------------------------
            local_path_dep = Path(DM_DATA_PATH, row['region']+'waves', row["datapath"])
            remote_path_dep = Path(ARCHIVE_BASE_PATH, row['region']+'waves', row['name'], 'delayedmode', row["datapath"])
            # remote_path_dep = Path(row["archive_path"]) 

            # local
            if not local_path_dep.exists():
                print(f"❌ [LOCAL/DEP] {local_path_dep}")
                continue
            
            # remote
            if not remote_path_dep.exists():
                print(f"❌ [IRDS/DEP] {remote_path_dep}")
                if remote_path_dep.parent.exists():
                    print(f"📁 [IRDS/DEP] Creating")
                    remote_path_dep.mkdir()
                elif not remote_path_dep.parent.parent.exists():
                    print(f"❌ [IRDS/SITE] {remote_path_dep}")
                    continue
            
            # RAW DATA (log) ------------------------------------------------
            local_raw_data = local_path_dep / RAWDATAFOLDER
            remote_raw_data = remote_path_dep / RAWDATAFOLDER

            # local
            if not local_raw_data.exists():
                print(f"❌ [LOCAL/DEP/RAW] {local_raw_data}")
                continue

            # remote
            if not remote_raw_data.exists():
                print(f"❌ [IRDS/DEP/RAW] {remote_raw_data}")
                
                if row["archive"] == "1":
                    print(f"▶ [COPYING/RAW DATA] {local_raw_data} → {remote_raw_data}")
                    copy_with_robocopy(local_raw_data, remote_raw_data)
                    print(f"✔️ Done")

            # PROCESSED DATA (processed_py) ------------------------------------------------
            local_proc_py = local_path_dep / SUBFOLDER
            remote_proc_py = remote_path_dep / SUBFOLDER

            # local
            if not local_proc_py.exists():
                print(f"❌ [LOCAL/DEP/PROCESS_PY] {local_proc_py}")
                continue

            # remote
            if remote_proc_py.exists():
                print(f"⚠️ [IRDS/DEP/PROCESS_PY] overwritting {remote_proc_py}")
            else:
                print(f"🟢 [IRDS/DEP/PROCESS_PY]")

            if row["archive"] == "1":
                print(f"▶ [COPYING/PROCESSED DATA] {local_proc_py} → {remote_proc_py}")
                copy_with_robocopy(local_proc_py, remote_proc_py)
                print(f"✔️ Done")

def copy_with_rsync(src, dst):
        
    import shutil

    print("rsync found at:", shutil.which("rsync"))

    
    cmd = [
        "rsync",
        "-avh",
        "--progress",
        "--partial",
        "--inplace",
        f"{src}/",
        f"{dst}/"
    ]
    subprocess.run(cmd, check=True)

def copy_with_robocopy(src, dst):
    cmd = [
        "robocopy",
        str(src),
        str(dst),
        "/E",      # copy subdirectories, including empty ones
        "/Z",      # restartable mode
        "/R:3",    # retry 3 times
        "/W:5",    # wait 5 seconds between retries
        "/NFL",    # no file list (cleaner output)
        "/NDL",    # no directory list
        "/NP",     # no progress per file (faster logs)
        "/XO",
        "/MT",
    ]

    result = subprocess.run(cmd)

    if result.returncode >= 8:
        raise RuntimeError(f"Robocopy failed with code {result.returncode}")


if __name__ == "__main__":
        
    main()