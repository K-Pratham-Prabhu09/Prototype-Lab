import sys
from pathlib import Path

# Ensure `utils/` is on sys.path so `from utils.GridWorld import GridWorld` works.
# This is needed because notebook kernels may start with a different working dir.
cwd = Path.cwd().resolve()
if (cwd / "utils").exists():
    repo_root = cwd
elif (cwd.parent / "utils").exists():
    repo_root = cwd.parent
else:
    raise RuntimeError("Could not locate utils/ directory relative to cwd")
utilsPath =str(repo_root / "utils")
sys.path.insert(0, utilsPath)
print(f" {utilsPath} Path addedd successfully")