"""
External Drive Reader - Read files directly from USB drives, external HDDs, 
or any mounted storage device without copying to local folder.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from processor.contentReader import ContentReader
import platform

class ExternalDriveReader(ContentReader):
    """Read documents directly from external drives and storage devices."""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        super().__init__(content_dir="", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.detected_drives = []
    
    def detect_drives(self) -> List[str]:
        """
        Automatically detect all available drives on the system.
        Works for Windows, macOS, and Linux.
        
        Returns:
            List of drive paths
        """
        system = platform.system()
        drives = []
        
        if system == "Windows":
            # Windows: Check all drive letters A-Z
            import string
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    drives.append(drive)
        
        elif system == "Darwin":  # macOS
            # macOS: Check /Volumes
            volumes_path = "/Volumes"
            if os.path.exists(volumes_path):
                for volume in os.listdir(volumes_path):
                    drive_path = os.path.join(volumes_path, volume)
                    if os.path.ismount(drive_path):
                        drives.append(drive_path)
        
        elif system == "Linux":
            # Linux: Check common mount points
            mount_points = ["/media", "/mnt", "/run/media"]
            for mount_base in mount_points:
                if os.path.exists(mount_base):
                    # Check user-specific mounts
                    for user_dir in os.listdir(mount_base):
                        user_path = os.path.join(mount_base, user_dir)
                        if os.path.isdir(user_path):
                            for mount in os.listdir(user_path):
                                drive_path = os.path.join(user_path, mount)
                                if os.path.ismount(drive_path):
                                    drives.append(drive_path)
        
        self.detected_drives = drives
        return drives
    
    def list_drives_interactive(self):
        """Display all detected drives and let user choose."""
        drives = self.detect_drives()
        
        if not drives:
            print("No external drives detected!")
            return None
        
        print("\n" + "=" * 60)
        print("Detected Drives:")
        print("=" * 60)
        
        for i, drive in enumerate(drives, 1):
            # Get drive info
            try:
                total, used, free = self._get_drive_space(drive)
                drive_name = os.path.basename(drive) or drive
                print(f"{i}. {drive}")
                print(f"   Name: {drive_name}")
                print(f"   Free Space: {self._format_bytes(free)}")
                print(f"   Total Space: {self._format_bytes(total)}")
            except:
                print(f"{i}. {drive}")
        
        print("=" * 60)
        
        while True:
            choice = input(f"\nSelect drive (1-{len(drives)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            try:
                index = int(choice) - 1
                if 0 <= index < len(drives):
                    return drives[index]
                else:
                    print(f"Please enter a number between 1 and {len(drives)}")
            except ValueError:
                print("Please enter a valid number")
    
    def browse_directory(self, base_path: str) -> Optional[str]:
        """
        Interactive directory browser for selecting a folder on the drive.
        
        Args:
            base_path: Starting path to browse from
            
        Returns:
            Selected directory path or None
        """
        current_path = base_path
        
        while True:
            print("\n" + "=" * 60)
            print(f"Current Location: {current_path}")
            print("=" * 60)
            
            try:
                items = []
                # List directories
                for item in os.listdir(current_path):
                    item_path = os.path.join(current_path, item)
                    if os.path.isdir(item_path):
                        items.append(("📁", item, item_path))
                
                # Sort directories
                items.sort(key=lambda x: x[1].lower())
                
                # Display options
                print("\nDirectories:")
                for i, (icon, name, path) in enumerate(items, 1):
                    print(f"{i}. {icon} {name}")
                
                print("\nOptions:")
                print("  [number] - Open directory")
                print("  'use' - Use current directory")
                print("  'back' - Go up one level")
                print("  'path' - Enter custom path")
                print("  'q' - Quit")
                
                choice = input("\nYour choice: ").strip().lower()
                
                if choice == 'q':
                    return None
                elif choice == 'use':
                    return current_path
                elif choice == 'back':
                    parent = os.path.dirname(current_path)
                    if parent != current_path:  # Not at root
                        current_path = parent
                elif choice == 'path':
                    custom_path = input("Enter full path: ").strip()
                    if os.path.exists(custom_path) and os.path.isdir(custom_path):
                        current_path = custom_path
                    else:
                        print("Invalid path!")
                else:
                    try:
                        index = int(choice) - 1
                        if 0 <= index < len(items):
                            current_path = items[index][2]
                    except ValueError:
                        print("Invalid choice!")
                
            except PermissionError:
                print("⚠ Permission denied! Try a different directory.")
            except Exception as e:
                print(f"Error: {e}")
    
    def read_from_drive(self, drive_path: str, 
                       subfolder: Optional[str] = None,
                       file_types: Optional[List[str]] = None,
                       split_docs: bool = True) -> List:
        """
        Read all documents from a drive or specific subfolder.
        
        Args:
            drive_path: Path to the drive (e.g., 'D:\\' or '/media/usb')
            subfolder: Optional subfolder within the drive
            file_types: Filter by file extensions (e.g., ['.pdf', '.docx'])
            split_docs: Whether to split documents into chunks
            
        Returns:
            List of Document objects
        """
        # Construct full path
        if subfolder:
            full_path = os.path.join(drive_path, subfolder)
        else:
            full_path = drive_path
        
        if not os.path.exists(full_path):
            raise ValueError(f"Path does not exist: {full_path}")
        
        if not os.path.isdir(full_path):
            raise ValueError(f"Path is not a directory: {full_path}")
        
        print("\n" + "=" * 60)
        print(f"Reading from: {full_path}")
        print("=" * 60)
        
        # Set the content directory temporarily
        self.content_dir = full_path
        
        # Get all files
        all_files = self.get_all_files()
        
        # Filter by file types if specified
        if file_types:
            all_files = [f for f in all_files if any(f.endswith(ext) for ext in file_types)]
        
        print(f"Found {len(all_files)} file(s) to process")
        
        # Read all files
        results = {}
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            print(f"\nReading: {file_name}")
            
            documents = self.read_single_file(file_path)
            if documents:
                if split_docs:
                    documents = self.text_splitter.split_documents(documents)
                results[file_path] = documents
                print(f"  ✓ Loaded {len(documents)} chunk(s)")
        
        # Combine all documents
        all_documents = []
        for docs in results.values():
            all_documents.extend(docs)
        
        print("\n" + "=" * 60)
        print(f"Total documents loaded: {len(all_documents)}")
        print("=" * 60)
        
        return all_documents
    
    def quick_scan(self, drive_path: str) -> Dict:
        """
        Quick scan to show file statistics without loading content.
        
        Args:
            drive_path: Path to scan
            
        Returns:
            Dictionary with file statistics
        """
        self.content_dir = drive_path
        all_files = self.get_all_files()
        
        stats = {
            'total_files': len(all_files),
            'by_extension': {},
            'total_size': 0
        }
        
        for file_path in all_files:
            ext = Path(file_path).suffix.lower()
            stats['by_extension'][ext] = stats['by_extension'].get(ext, 0) + 1
            
            try:
                stats['total_size'] += os.path.getsize(file_path)
            except:
                pass
        
        return stats
    
    def _get_drive_space(self, path: str):
        """Get drive space information."""
        import shutil
        total, used, free = shutil.disk_usage(path)
        return total, used, free
    
    def _format_bytes(self, bytes_size: int) -> str:
        """Format bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} PB"


def interactive_drive_selection():
    """Interactive mode to select and read from external drive."""
    reader = ExternalDriveReader()
    
    print("\n" + "=" * 60)
    print("External Drive Document Reader")
    print("=" * 60)
    
    # Step 1: Detect and select drive
    selected_drive = reader.list_drives_interactive()
    if not selected_drive:
        print("No drive selected. Exiting.")
        return None
    
    print(f"\n✓ Selected drive: {selected_drive}")
    
    # Step 2: Browse for folder (optional)
    print("\nWould you like to:")
    print("1. Read entire drive")
    print("2. Browse and select a specific folder")
    
    choice = input("\nChoice (1 or 2): ").strip()
    
    if choice == '2':
        selected_folder = reader.browse_directory(selected_drive)
        if not selected_folder:
            print("No folder selected. Exiting.")
            return None
        read_path = selected_folder
    else:
        read_path = selected_drive
    
    # Step 3: File type filter (optional)
    print("\nFilter by file types?")
    print("1. Read all supported files")
    print("2. Specify file types (e.g., .pdf,.docx)")
    
    choice = input("\nChoice (1 or 2): ").strip()
    file_types = None
    
    if choice == '2':
        types_input = input("Enter file extensions (comma-separated, e.g., .pdf,.docx,.txt): ").strip()
        file_types = [t.strip() for t in types_input.split(',')]
        print(f"Filtering for: {file_types}")
    
    # Step 4: Quick scan
    print("\nPerforming quick scan...")
    stats = reader.quick_scan(read_path)
    print(f"\nFound {stats['total_files']} file(s)")
    print(f"Total size: {reader._format_bytes(stats['total_size'])}")
    print("\nFile types:")
    for ext, count in stats['by_extension'].items():
        print(f"  {ext}: {count} file(s)")
    
    # Step 5: Confirm and read
    confirm = input("\nProceed with reading these files? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Operation cancelled.")
        return None
    
    # Step 6: Read documents
    documents = reader.read_from_drive(
        drive_path=read_path,
        file_types=file_types,
        split_docs=True
    )
    
    return documents


# Example usage
if __name__ == "__main__":
    # Method 1: Interactive mode (recommended)
    print("=== INTERACTIVE MODE ===")
    documents = interactive_drive_selection()
    
    # Method 2: Direct path (if you know the exact path)
    # reader = ExternalDriveReader()
    # 
    # # Windows example
    # documents = reader.read_from_drive("E:\\", subfolder="Documents")
    # 
    # # macOS example
    # documents = reader.read_from_drive("/Volumes/MyUSB", subfolder="PDFs")
    # 
    # # Linux example
    # documents = reader.read_from_drive("/media/user/USB_DRIVE", file_types=['.pdf', '.txt'])