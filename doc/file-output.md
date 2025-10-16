# File Output Guide

This guide explains how to run the Hailo RPi5 examples and save the output to files, specifically to the `/tmp` directory.

## Prerequisites

Before running any examples, ensure your environment is properly configured:

```bash
source setup_env.sh
```

This script activates the virtual environment and sets required environment variables.

## Available Examples

The `basic_pipelines/` directory contains several examples:

- `detection.py` - Full object detection with tracking
- `detection_simple.py` - Lightweight detection example
- `instance_segmentation.py` - Instance segmentation
- `pose_estimation.py` - Human pose estimation
- `depth.py` - Depth estimation

## Standard Command-Line Options

All examples support the following command-line arguments:

| Option | Description |
|--------|-------------|
| `--input`, `-i` | Input source (file path, `rpi` for RPi camera, `usb` for USB camera, `/dev/video<X>` for specific camera) |
| `--use-frame`, `-u` | Enable frame processing in callback function |
| `--show-fps`, `-f` | Display FPS counter on video output |
| `--arch` | Specify Hailo architecture (`hailo8`, `hailo8l`, `hailo10h`) |
| `--hef-path` | Path to custom HEF model file |
| `--disable-sync` | Run as fast as possible (useful for file input) |
| `--disable-callback` | Disable custom callback function |
| `--dump-dot` | Export pipeline graph to `pipeline.dot` file |
| `--frame-rate`, `-r` | Set frame rate (default: 30) |

## Running Examples with Display Output

### Basic execution with default video:
```bash
python basic_pipelines/detection.py
```

### With Raspberry Pi camera:
```bash
python basic_pipelines/detection.py --input rpi
```

### With USB webcam (auto-detect):
```bash
python basic_pipelines/detection.py --input usb
```

### With video file:
```bash
python basic_pipelines/detection.py --input /path/to/video.mp4
```

### Show FPS counter:
```bash
python basic_pipelines/detection.py --show-fps
```

To close any example, press `Ctrl+C`.

## Saving Output to Files

### Method 1: Modify Pipeline to Use FILE_SINK_PIPELINE

The Hailo Apps Infrastructure provides a `FILE_SINK_PIPELINE` function that can save video output. Here's how to create a custom script that saves to `/tmp`:

**Create a new file** (e.g., `basic_pipelines/detection_with_output.py`):

```python
from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class, dummy_callback
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE, INFERENCE_PIPELINE_WRAPPER, INFERENCE_PIPELINE,
    TRACKER_PIPELINE, USER_CALLBACK_PIPELINE, FILE_SINK_PIPELINE, OVERLAY_PIPELINE, QUEUE
)
from hailo_apps.hailo_app_python.core.common.core import get_default_parser, get_resource_path
from hailo_apps.hailo_app_python.core.common.defines import (
    DETECTION_PIPELINE, RESOURCES_MODELS_DIR_NAME, RESOURCES_SO_DIR_NAME,
    DETECTION_POSTPROCESS_SO_FILENAME, DETECTION_POSTPROCESS_FUNCTION
)
from hailo_apps.hailo_app_python.core.common.installation_utils import detect_hailo_arch
import setproctitle

class DetectionWithFileOutput(GStreamerDetectionApp):
    def __init__(self, app_callback, user_data, output_path="/tmp/output.mkv", parser=None):
        self.output_path = output_path
        super().__init__(app_callback, user_data, parser)

    def get_pipeline_string(self):
        """Override to use FILE_SINK_PIPELINE instead of DISPLAY_PIPELINE"""
        source_pipeline = SOURCE_PIPELINE(
            video_source=self.video_source,
            video_width=self.video_width,
            video_height=self.video_height,
            frame_rate=self.frame_rate,
            sync=self.sync
        )

        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=self.batch_size,
            config_json=self.labels_json,
            additional_params=self.thresholds_str
        )

        detection_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(detection_pipeline)
        tracker_pipeline = TRACKER_PIPELINE(class_id=1)
        user_callback_pipeline = USER_CALLBACK_PIPELINE()

        # Use OVERLAY_PIPELINE to draw detections, then FILE_SINK_PIPELINE to save
        overlay_pipeline = OVERLAY_PIPELINE()
        file_sink_pipeline = FILE_SINK_PIPELINE(output_file=self.output_path, bitrate=5000)

        pipeline_string = (
            f'{source_pipeline} ! '
            f'{detection_pipeline_wrapper} ! '
            f'{tracker_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{overlay_pipeline} ! '
            f'{file_sink_pipeline}'
        )

        print(f"Saving output to: {self.output_path}")
        print(pipeline_string)
        return pipeline_string

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)

    # Parse command line arguments
    parser = get_default_parser()
    parser.add_argument("--output", "-o", type=str, default="/tmp/output.mkv",
                       help="Output file path (default: /tmp/output.mkv)")
    parser.add_argument("--labels-json", default=None, help="Path to custom labels JSON file")

    args = parser.parse_args()

    # Create app with file output
    user_data = app_callback_class()
    app = DetectionWithFileOutput(dummy_callback, user_data,
                                  output_path=args.output, parser=parser)
    app.run()

    print(f"\nRecording complete! Output saved to: {args.output}")
    print("Note: You may need to fix the file header with ffmpeg:")
    print(f"  ffmpeg -i {args.output} -c copy {args.output.replace('.mkv', '_fixed.mkv')}")
```

**Run the script:**
```bash
# Save to /tmp with default filename
python basic_pipelines/detection_with_output.py --input /path/to/video.mp4

# Specify custom output path
python basic_pipelines/detection_with_output.py --input rpi --output /tmp/my_detection.mkv

# Run without sync for faster processing
python basic_pipelines/detection_with_output.py --input /path/to/video.mp4 --disable-sync --output /tmp/output.mkv
```

### Method 2: Use Callback to Save Frames

You can also save individual frames or create your own video writer in the callback function:

```python
from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import os
import cv2
import hailo

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

class VideoWriterCallback(app_callback_class):
    def __init__(self, output_path="/tmp/output.mp4"):
        super().__init__()
        self.video_writer = None
        self.output_path = output_path
        self.frame_count = 0

    def init_writer(self, width, height, fps=30):
        """Initialize video writer"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_path, fourcc, fps, (width, height)
        )
        print(f"Initialized video writer: {self.output_path}")

    def cleanup(self):
        """Release video writer"""
        if self.video_writer:
            self.video_writer.release()
            print(f"Saved {self.frame_count} frames to {self.output_path}")

def save_frames_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Get frame dimensions
    format, width, height = get_caps_from_pad(pad)

    # Initialize writer on first frame
    if user_data.video_writer is None and format and width and height:
        user_data.init_writer(width, height)

    # Get frame data
    if user_data.use_frame and format and width and height:
        frame = get_numpy_from_buffer(buffer, format, width, height)

        # Get detections and draw on frame
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        for detection in detections:
            label = detection.get_label()
            bbox = detection.get_bbox()
            confidence = detection.get_confidence()

            # Draw bounding box
            x, y, w, h = int(bbox.xmin() * width), int(bbox.ymin() * height), \
                         int(bbox.width() * width), int(bbox.height() * height)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}",
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write frame
        if user_data.video_writer:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            user_data.video_writer.write(frame_bgr)
            user_data.frame_count += 1

            if user_data.frame_count % 30 == 0:
                print(f"Processed {user_data.frame_count} frames")

    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)

    # Create callback with video writer
    user_data = VideoWriterCallback(output_path="/tmp/detection_output.mp4")
    user_data.use_frame = True  # Must enable frame processing

    app = GStreamerDetectionApp(save_frames_callback, user_data)

    try:
        app.run()
    finally:
        user_data.cleanup()
```

## Post-Processing Output Files

### Fix MKV file headers:
When using `FILE_SINK_PIPELINE`, the output MKV file may have incomplete headers. Fix with:

```bash
ffmpeg -i /tmp/output.mkv -c copy /tmp/output_fixed.mkv
```

### Convert to different formats:
```bash
# Convert to MP4
ffmpeg -i /tmp/output.mkv -c:v libx264 -c:a aac /tmp/output.mp4

# Extract frames as images
ffmpeg -i /tmp/output.mkv /tmp/frame_%04d.png

# Reduce file size
ffmpeg -i /tmp/output.mkv -vcodec libx264 -crf 28 /tmp/output_compressed.mp4
```

## Saving Individual Frames

To save individual frames instead of video:

```python
# In your callback function:
def save_frame_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    frame_num = user_data.get_count()

    format, width, height = get_caps_from_pad(pad)

    if user_data.use_frame and format and width and height:
        frame = get_numpy_from_buffer(buffer, format, width, height)

        # Save every 30th frame
        if frame_num % 30 == 0:
            output_path = f"/tmp/frame_{frame_num:06d}.jpg"
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, frame_bgr)
            print(f"Saved: {output_path}")

    return Gst.PadProbeReturn.OK
```

## Tips and Best Practices

1. **Storage Space**: The `/tmp` directory is typically stored in RAM. For long recordings, use a different directory:
   ```bash
   python basic_pipelines/detection_with_output.py --output ~/Videos/output.mkv
   ```

2. **Performance**: Use `--disable-sync` when processing video files to run as fast as possible:
   ```bash
   python basic_pipelines/detection.py --input video.mp4 --disable-sync
   ```

3. **Bitrate Control**: Adjust bitrate in `FILE_SINK_PIPELINE` for quality vs file size:
   ```python
   file_sink_pipeline = FILE_SINK_PIPELINE(output_file=path, bitrate=8000)  # Higher quality
   ```

4. **Monitor Disk Space**: Check available space before long recordings:
   ```bash
   df -h /tmp
   ```

5. **Clean Up**: Remove old files from `/tmp` regularly:
   ```bash
   rm /tmp/output*.mkv
   ```

## Troubleshooting

### "No space left on device"
- `/tmp` is limited in size. Use a different output directory or increase tmpfs size.

### Output file is corrupted
- Always use ffmpeg to fix MKV headers after recording.
- Ensure the application completes gracefully (wait for Ctrl+C to finish).

### Pipeline fails to start
- Ensure environment is sourced: `source setup_env.sh`
- Check that input source exists and is accessible.
- Verify write permissions for output directory.

### Choppy/dropped frames in output
- Reduce video resolution
- Increase bitrate
- Use `--disable-callback` to reduce CPU load
- Check CPU usage with `htop`

## Additional Resources

- [Basic Pipelines Documentation](basic-pipelines.md)
- [Hailo Apps Infra Repository](https://github.com/hailo-ai/hailo-apps-infra)
- [GStreamer Documentation](https://gstreamer.freedesktop.org/documentation/)
