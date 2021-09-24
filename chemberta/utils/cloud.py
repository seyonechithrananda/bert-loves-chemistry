import subprocess


def check_cloud(path: str):
    """Naive check to if the path is a cloud path"""
    if path.startswith("s3:"):
        return True
    return False


def sync_with_s3(source_dir: str, target_dir: str):
    """Sync source_dir directory with target_dir"""
    subprocess.check_call(
        [
            "aws",
            "s3",
            "sync",
            source_dir,
            target_dir,
            "--acl",
            "bucket-owner-full-control",
        ]
    )
    return
