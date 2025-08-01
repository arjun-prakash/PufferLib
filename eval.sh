#!/bin/bash
set -euo pipefail

# Config
DISPLAY_NUM=":99"
SCREEN="0"
RESOLUTION="1920x1080x24"   # Higher internal resolution
VIDEO_FPS=15
OUTPUT_GIF="my_eval.gif"

# Start Xvfb
Xvfb ${DISPLAY_NUM} -screen ${SCREEN} ${RESOLUTION} &
XVFB_PID=$!
export DISPLAY=${DISPLAY_NUM}

# Ensure cleanup on exit
cleanup() {
  echo "Cleaning up Xvfb..."
  kill ${XVFB_PID} 2>/dev/null || true
}
trap cleanup EXIT

# Run eval (this will produce frame_*.png via TakeScreenshot)
python -m pufferlib.pufferl eval puffer_drone_crazyflie \
  --load-model-path latest \
  --gif-path dummy.gif \
  --render-mode human --save-frames 100

# Assemble GIF at higher resolution (assuming frame_*.png are large already)
ffmpeg -framerate ${VIDEO_FPS} -i frame_%06d.png -vf "scale=1920:-1:flags=lanczos,palettegen" palette.png
ffmpeg -framerate ${VIDEO_FPS} -i frame_%06d.png -i palette.png \
  -filter_complex "scale=1920:-1:flags=lanczos[x];[x][1:v]paletteuse" "${OUTPUT_GIF}"

# Optional optimize if gifsicle exists
if command -v gifsicle >/dev/null 2>&1; then
  gifsicle -O3 --colors 128 "${OUTPUT_GIF}" -o optimized_${OUTPUT_GIF}
  echo "Optimized GIF: optimized_${OUTPUT_GIF}"
fi

# Cleanup frames and palette
rm frame_*.png palette.png

echo "Done. GIF: ${OUTPUT_GIF}"
