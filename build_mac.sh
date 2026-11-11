#!/bin/bash
# Builds the macOS app "dist/t-DiFC Live.app" from tdifc_app_desktop.py and a zip to share it:
#   dist/t-DiFC-Live-mac-<arch>.zip
# Run in this folder with the project's environment set up (README: SETUP):
#   ./build_mac.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=t-difc-rt-env/bin/python
APP="t-DiFC Live"
ARCH=$(uname -m)                                      # arm64: Apple silicon Macs only

"$PY" -m pip install -q -r requirements-build.txt
"$PY" -m PyInstaller --noconfirm --clean tdifc_app_desktop.spec

# ditto keeps the app's symbolic links and signature (a plain zip can break the app)
rm -f "dist/t-DiFC-Live-mac-$ARCH.zip"
ditto -c -k --sequesterRsrc --keepParent "dist/$APP.app" "dist/t-DiFC-Live-mac-$ARCH.zip"
echo
echo "app: dist/$APP.app ($(du -sh "dist/$APP.app" | cut -f1))"
echo "zip: dist/t-DiFC-Live-mac-$ARCH.zip ($(du -sh "dist/t-DiFC-Live-mac-$ARCH.zip" | cut -f1)) - share this file"
