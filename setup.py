import sys
import subprocess
import pkg_resources

required = {'numpy', 'pandas', 'matplotlib', 'scipy', 'tqdm'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

for pkg in missing:
    print(f"Installing module {pkg}...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
print("All required package installed")