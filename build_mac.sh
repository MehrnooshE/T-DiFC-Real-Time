#!/bin/bash
# Builds the macOS app "dist/t-DiFC Live v3.app" from tdifc_app_desktop.py and a zip to share it:
#   dist/t-DiFC-Live-v3-mac-<arch>.zip
# Run in this folder with the project's environment set up (README: SETUP):
#   ./build_mac.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=t-difc-rt-env/bin/python
APP="t-DiFC Live v3"
ARCH=$(uname -m)                                      # arm64: Apple silicon Macs only
ZIP="dist/t-DiFC-Live-v3-mac-$ARCH.zip"

"$PY" -m pip install -q -r requirements-build.txt
"$PY" -m PyInstaller --noconfirm --clean tdifc_app_desktop.spec

# ditto keeps the app's symbolic links and signature (a plain zip can break the app)
rm -f "$ZIP"
ditto -c -k --sequesterRsrc --keepParent "dist/$APP.app" "$ZIP"
echo
echo "app: dist/$APP.app ($(du -sh "dist/$APP.app" | cut -f1))"
echo "zip: $ZIP ($(du -sh "$ZIP" | cut -f1)) - share this file"
