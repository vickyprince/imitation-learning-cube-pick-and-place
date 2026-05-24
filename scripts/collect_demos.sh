#!/bin/bash
# Usage: ./scripts/collect_demos.sh [n_demos]
N=${1:-5}
echo "Collecting $N demonstrations via webserver teleoperation."
echo "Open http://localhost:9000 and use the Console panel."
echo ""
echo "  1. Use the joystick to position the arm over the cube."
echo "  2. Press 'Start Demo' button to begin recording."
echo "  3. Perform the pick-and-place."
echo "  4. Press 'Stop Demo' to save. Press 'Discard' if the demo was bad."
echo "  5. Repeat $N times."
echo ""
echo "Bags will be saved to: ./data/bags/"
