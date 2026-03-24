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
                    remote_path_dep.mkdir()
                    print(f"📁 [IRDS/DEP] created")
                elif not remote_path_dep.parent.parent.exists():
                    print(f"❌ [IRDS/SITE] {remote_path_dep}")
                    continue
            else:
                print(f"🟢 [IRDS/DEP] {remote_path_dep}")

            # RAW DATA (log) ------------------------------------------------
            local_raw_data = local_path_dep / RAWDATAFOLDER
            remote_raw_data = remote_path_dep / RAWDATAFOLDER

            # local
            if local_raw_data.exists():
                print(f"🟢 [LOCAL/DEP/RAW] {local_raw_data}")
            else:   
                print(f"❌ [LOCAL/DEP/RAW] {local_raw_data}")
                continue

            # remote
            if not remote_raw_data.exists():
                print(f"❌ [IRDS/DEP/RAW] {remote_raw_data}")
                
                if row["archive"] == "1":
                    print(f"🔄 [COPYING/RAW DATA] {local_raw_data} → {remote_raw_data}")
                    copy_with_robocopy(local_raw_data, remote_raw_data)
                    print(f"✔️ [IRDS/DEP/RAW] copied")

            else:
                print(f"🟢 [IRDS/DEP/RAW] {remote_raw_data}")

            # PROCESSED DATA (processed_py) ------------------------------------------------
            local_proc_py = local_path_dep / SUBFOLDER
            remote_proc_py = remote_path_dep / SUBFOLDER

            # local
            if not local_proc_py.exists():
                print(f"❌ [LOCAL/DEP/PROCESS_PY] {local_proc_py}")
                continue

            # remote
            if not remote_proc_py.exists():
                print(f"❌ [IRDS/DEP/PROCESS_PY] {remote_proc_py}")
            else:
                print(f"🟢 [IRDS/DEP/PROCESS_PY] {remote_proc_py}")
                print(f"⚠️ [IRDS/DEP/PROCESS_PY] overwritting {remote_proc_py}")
            if row["archive"] == "1":
                print(f"🔄 [COPYING/PROCESSED DATA] {local_proc_py} → {remote_proc_py}")
                copy_with_robocopy(local_proc_py, remote_proc_py)
                print(f"✔️ [IRDS/DEP/PROCESS_PY] copied")


            
            # DELETE LOCAL FOLDER ------------------------------------------------
            # ARCHIVE_BASE = Path(DM_DATA_PATH, row['region'] + 'waves', 'archived')
            # archived_folder = ARCHIVE_BASE / row["datapath"]
            # archived_folder.mkdir(parents=True, exist_ok=True)

            # delete_local_folder_with_robocopy(local_path_dep)
            # move_to_archived(local_path_dep, ARCHIVE_BASE)
            # print(f"🟢 [LOCAL/DEP] archived, locally deleted and moved to 'archived' folder: {local_path_dep}")

            # DELETE LOCAL FOLDER ------------------------------------------------
            ARCHIVE_BASE = Path(DM_DATA_PATH, row['region'] + 'waves')

            delete_local_folder_with_robocopy(local_path_dep)

            log_archived(row["datapath"], ARCHIVE_BASE)

            print(f"✔️ [LOCAL/DEP] archived and logged: {local_path_dep}")



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

def delete_local_folder_with_robocopy(folder_path:Path):
    
    path = Path(folder_path)  # ensure Path object
    if not path.exists():
        raise FileNotFoundError(f"Folder does not exist: {path}")
    
    # Delete folder and all contents quietly
    subprocess.run(f'cmd /c "rd /s /q "{path}""', shell=True, check=True)
    
    # # Recreate empty folder
    # path.mkdir(parents=True, exist_ok=True)

def move_to_archived(source_folder: Path, archived_base: Path):

    source_folder = Path(source_folder)
    archived_base = Path(archived_base)
    
    if not source_folder.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_folder}")
    
    # Destination folder path
    dest_folder = archived_base / source_folder.name
    dest_folder.parent.mkdir(parents=True, exist_ok=True)
    
    # Move folder
    import shutil
    shutil.move(str(source_folder), str(dest_folder))
    
    return dest_folder

def log_archived(datapath: str, archive_base: Path):

    archive_base = Path(archive_base)
    # archive_base.mkdir(parents=True, exist_ok=True)

    archive_file = archive_base / "archived.txt"
    
    from datetime import datetime 
    with open(archive_file, "a", encoding="utf-8") as f:
        f.write(f"archived at: {datetime.now()} - {datapath}\n")


if __name__ == "__main__":
        
    main()