#!/usr/bin/env python3
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path


def create_submission():
    print("\n📦 Creating STRICT JIT submission package...")

    # EXACT WHITELIST
    required_files = ['solution.py', 'utils.py', 'model_jit.pt']
    submissions_dir = Path('submissions')
    submissions_dir.mkdir(exist_ok=True)
    zip_path = submissions_dir / 'submission.zip'

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for file_name in required_files:
            src_path = Path(file_name)
            if not src_path.exists():
                print(f"❌ ERROR: Required file '{file_name}' not found!")
                return False

            shutil.copy2(src_path, temp_path / file_name)
            print(f"Added: {file_name}")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_name in required_files:
                zipf.write(temp_path / file_name, arcname=file_name)

    print(f"\n✅ Done! {zip_path} is ready for upload.")
    return True


if __name__ == "__main__":
    success = create_submission()
    sys.exit(0 if success else 1)