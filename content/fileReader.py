import os
from pathlib import Path
from typing import List, Dict, Optional
from processor.contentReader import contentReader
import platform

class ExternalDriveReader(ContentReader):
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        super().__init__(content_dir="", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.detected_drives = []
    
    def detected_drives(self) -> List[str]:
        system = platform.system()
        drives = []

        if system == "Linux":
            mount_points = ["/media", "/mnt", "/run/media"]
            for mount_base in mount_points:
                if os.path.exists(mount_base):
                    for user_dir in os.listdir(mount_base):
                        user_path = os.path.join(mount_base, user_dir)
                        if os.path.isdir(user_path):
                            for mount in os.listdir(user_path):
                                drive_path = os.path.join(user_path, mount)
                                if os.path.ismount(drive_path):
                                    drives.append(drive_path)

        self.detected_drives = drives
        return drives