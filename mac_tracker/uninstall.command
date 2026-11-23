#!/bin/bash
# Removes Teleios Tracker's daily schedule. Your data in ~/TeleiosTracker is
# kept; drag that folder to the Trash if you want it gone too.
LABEL="com.teleios.tracker"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/$LABEL.plist"
echo "Teleios Tracker's daily run is removed."
echo "The folder ~/TeleiosTracker is still there; delete it yourself if you like."
read -r -p "Press Return to close."
