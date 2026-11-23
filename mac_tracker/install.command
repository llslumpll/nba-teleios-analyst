#!/bin/bash
# Teleios Tracker installer. Double-click to run.
# What it does (nothing else):
#   1. copies the tracker into ~/TeleiosTracker (outside Documents, so the
#      daily job doesn't need extra macOS privacy permissions)
#   2. makes a private Python environment there and installs nba_api
#   3. runs one test pull
#   4. schedules a daily run at 9:05 AM (runs on wake if the Mac was asleep)
# No admin password, no system settings. Undo with uninstall.command.

set -e
SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="$HOME/TeleiosTracker"
LABEL="com.teleios.tracker"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

echo "== Teleios Tracker setup =="
mkdir -p "$DEST/logs"
cp "$SRC/tracker.py" "$SRC/uninstall.command" "$DEST/"
chmod +x "$DEST/uninstall.command"

PY=""
for c in "$HOME/anaconda3/bin/python3" "$HOME/opt/anaconda3/bin/python3" "/opt/anaconda3/bin/python3" "/opt/homebrew/bin/python3" "/usr/local/bin/python3" "/usr/bin/python3"; do
  if [ -x "$c" ] && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then echo "No Python 3.9+ found. Tell Claude: 'installer found no Python'."; read -r -p "Press Return to close."; exit 1; fi
echo "Using Python: $PY"

if [ ! -x "$DEST/.venv/bin/python" ]; then "$PY" -m venv "$DEST/.venv"; fi
"$DEST/.venv/bin/python" -m pip install --quiet --upgrade pip
"$DEST/.venv/bin/python" -m pip install --quiet "nba_api>=1.10" pandas
echo "Installed nba_api."

echo ""
echo "== Test pull from NBA.com (about 1-2 minutes) =="
"$DEST/.venv/bin/python" "$DEST/tracker.py" | tee "$DEST/logs/install_test.log" || true

mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$DEST/.venv/bin/python</string>
    <string>$DEST/tracker.py</string>
  </array>
  <key>WorkingDirectory</key><string>$DEST</string>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>9</integer><key>Minute</key><integer>5</integer></dict>
  <key>StandardOutPath</key><string>$DEST/logs/daily.log</string>
  <key>StandardErrorPath</key><string>$DEST/logs/daily.log</string>
  <key>ProcessType</key><string>Background</string>
</dict>
</plist>
EOF
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo ""
echo "== Done =="
echo "Daily run scheduled for 9:05 AM. Files are in: $DEST"
echo "You can close this window."
