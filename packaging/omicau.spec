# PyInstaller spec for omicau — onedir, CPU-only torch.
#
# Build:  pyinstaller packaging/omicau.spec --noconfirm --clean
# Never onefile (torch would extract hundreds of MB to temp on every launch and
# trip antivirus). No UPX (it corrupts torch shared libraries). torchvision /
# torchaudio are excluded; install CPU-only torch before building:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# The UI's static assets (index.html / app.js / ui.css) must ship with the app.
here = Path(SPECPATH).resolve()
pkg = here.parent / "omicau"
datas = [(str(pkg / "ui" / "static"), "omicau/ui/static")]
datas += collect_data_files("plotly")          # plotly bundles JSON/JS data
datas += collect_data_files("omicau")

# uvicorn + the optional FastAPI UI load submodules dynamically.
hiddenimports = (
    collect_submodules("uvicorn")
    + collect_submodules("omicau")
    + ["omicau.ui.routes", "omicau.ui.inspect", "omicau.ui.server",
       "anyio", "sklearn.utils._typedefs", "sklearn.neighbors._partition_nodes"]
)

excludes = ["torchvision", "torchaudio", "tkinter", "matplotlib", "notebook",
            "IPython", "pytest", "jupyter", "PyQt5", "PySide6", "test"]

a = Analysis(
    [str(here / "omicau_launcher.py")],
    pathex=[str(here.parent)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="omicau",
    console=True,                 # a console app; the UI opens a browser window
    disable_windowed_traceback=False,
    icon=None,
    upx=False,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False, name="omicau",
)
