# PyInstaller spec of the macOS app "t-DiFC Live.app" (tdifc_app_desktop.py). Build with ./build_mac.sh.
# One folder bundle (a one-file .app starts slower and macOS handles it badly); the model is bundled.
# -*- mode: python ; coding: utf-8 -*-

APP = "t-DiFC Live"
VERSION = "1.0.0"

a = Analysis(
    ["tdifc_app_desktop.py"],
    pathex=[],
    binaries=[],
    datas=[("models/fusion_lr1e-4_wd_ls_torch.pt", "models")],   # paths.MODEL_PATH inside the bundle
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "jupyter", "notebook", "sklearn", "tensorflow",
              "PyQt5", "PySide2", "PySide6", "torch.utils.tensorboard", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP,
    console=False,              # a window app, no terminal
    argv_emulation=False,
    target_arch=None,           # the build machine's (arm64 on Apple silicon)
    codesign_identity=None,     # ad-hoc signature; see README "Sharing the app" for the first start
    entitlements_file=None,
)
coll = COLLECT(exe, a.binaries, a.datas, name=APP)
app = BUNDLE(
    coll,
    name=f"{APP}.app",
    icon=None,
    bundle_identifier="edu.northeastern.niedrelab.tdifc-live",
    version=VERSION,
    info_plist={
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
        "NSRequiresAquaSystemAppearance": True,   # light window frame, as the app's light theme
    },
)
