from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import threading
import queue
import time
import setproctitle

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class, GStreamerApp
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE, INFERENCE_PIPELINE_WRAPPER, INFERENCE_PIPELINE,
    TRACKER_PIPELINE, USER_CALLBACK_PIPELINE, OVERLAY_PIPELINE, DISPLAY_PIPELINE, QUEUE
)
from hailo_apps.hailo_app_python.core.common.core import get_default_parser, get_resource_path
from hailo_apps.hailo_app_python.core.common.defines import (
    FACE_DETECTION_PIPELINE, RESOURCES_MODELS_DIR_NAME, RESOURCES_SO_DIR_NAME,
    FACE_DETECTION_POSTPROCESS_SO_FILENAME, RESOURCES_JSON_DIR_NAME, FACE_DETECTION_JSON_NAME
)
from hailo_apps.hailo_app_python.core.common.installation_utils import detect_hailo_arch

# -----------------------------------------------------------------------------------------------
# Face Detection App with Frame File Output (REAL-TIME with Async Writing)
# -----------------------------------------------------------------------------------------------
# This app uses SCRFD face detection models to detect faces and write annotated frames to files.
#
# Key Features:
# - Face detection using SCRFD (Sample and Computation Redistributed Face Detection)
# - Bounding boxes: All face detections are drawn with confidence scores
# - Asynchronous processing: Frame copying happens in callback (~2ms), all drawing and
#   file I/O happens in a background thread to avoid blocking the inference pipeline
# - Queue size: 30 frames buffered. If the queue fills up, frames are dropped instead of
#   blocking the pipeline.
# - True real-time: Pipeline runs at camera framerate (30fps) regardless of disk speed
# - Latency measurement: Tracks end-to-end latency from capture to disk write
#
# Three write modes are supported:
# 1. Overwrite mode (default): Continuously overwrites a single file with the latest frame
# 2. Write-all mode (--write-all): Writes each frame to a numbered file (frame_0001.jpg, etc.)
# 3. Copy-all mode (--copy-all): Renames previous frame to numbered backup before writing new frame
#    (output.jpg is the latest, output_0001.jpg, output_0002.jpg are previous frames)
# -----------------------------------------------------------------------------------------------

class AsyncFrameWriter:
    """Background thread for asynchronous file writing."""
    def __init__(self, max_queue_size=30):
        self.write_queue = queue.Queue(maxsize=max_queue_size)
        self.running = True
        self.thread = threading.Thread(target=self._writer_thread, daemon=True)
        self.frames_dropped = 0

        # Latency tracking
        self.latencies = []  # Store all latencies for analysis
        self.min_latency = float('inf')
        self.max_latency = 0.0
        self.total_latency = 0.0
        self.latency_count = 0

        self.thread.start()

    def _writer_thread(self):
        """Background thread that processes and writes frames to disk."""
        while self.running:
            try:
                # Get frame with timeout
                item = self.write_queue.get(timeout=0.1)
                if item is None:  # Shutdown signal
                    break

                # Record when we start processing this frame
                processing_start = time.time()

                # Handle both old and new format for backwards compatibility
                if isinstance(item, dict):
                    # New format with rename support
                    output_path = item['output_file']
                    frame_rgb = item['frame_rgb']
                    frame_count = item['frame_count']
                    detection_count = item['detection_count']
                    detections = item['detections']
                    rename_info = item.get('rename_info')
                    capture_timestamp = item.get('capture_timestamp', processing_start)
                else:
                    # Old format (for compatibility)
                    output_path, frame_rgb, frame_count, detection_count, detections = item
                    rename_info = None
                    capture_timestamp = processing_start

                # Handle rename operation (copy-all mode)
                if rename_info:
                    rename_from, rename_to = rename_info
                    try:
                        if Path(rename_from).exists():
                            import shutil
                            shutil.move(str(rename_from), str(rename_to))
                    except Exception as e:
                        print(f"Warning: Failed to rename {rename_from} to {rename_to}: {e}")

                # ALL processing happens here in background thread
                # Make a copy (in background, not blocking main pipeline)
                frame_copy = frame_rgb.copy()
                height, width = frame_copy.shape[:2]

                # Draw bounding boxes for all face detections
                for det in detections:
                    label, confidence, bbox = det

                    # Convert normalized bbox to pixel coordinates
                    x = int(bbox['xmin'] * width)
                    y = int(bbox['ymin'] * height)
                    w = int(bbox['width'] * width)
                    h = int(bbox['height'] * height)

                    # Draw bounding box (green for faces)
                    cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Draw label with confidence
                    label_text = f"Face: {confidence:.2f}"
                    cv2.putText(frame_copy, label_text,
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Add frame info overlays
                cv2.putText(frame_copy, f"Frame: {frame_count}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame_copy, f"Faces: {detection_count}",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Convert to BGR
                frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)

                # Write to disk
                cv2.imwrite(str(output_path), frame_bgr)

                # Calculate total latency (capture to write complete)
                write_complete = time.time()
                total_latency = write_complete - capture_timestamp
                queue_wait = processing_start - capture_timestamp
                processing_time = write_complete - processing_start

                # Update latency statistics
                self.latencies.append(total_latency)
                self.min_latency = min(self.min_latency, total_latency)
                self.max_latency = max(self.max_latency, total_latency)
                self.total_latency += total_latency
                self.latency_count += 1

                # Periodic latency reporting (every 100 frames)
                if self.latency_count % 100 == 0:
                    avg_latency = self.total_latency / self.latency_count
                    print(f"[Latency] Frame {frame_count}: "
                          f"Total={total_latency*1000:.1f}ms "
                          f"(Queue={queue_wait*1000:.1f}ms + Process={processing_time*1000:.1f}ms) | "
                          f"Avg={avg_latency*1000:.1f}ms Min={self.min_latency*1000:.1f}ms Max={self.max_latency*1000:.1f}ms")

                self.write_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in writer thread: {e}")

    def write_async_with_rename(self, write_info):
        """Queue a frame for asynchronous processing and writing (with optional rename)."""
        try:
            # Non-blocking put - pass the dict containing all info
            self.write_queue.put_nowait(write_info)
            return True
        except queue.Full:
            self.frames_dropped += 1
            return False

    def shutdown(self):
        """Shutdown the writer thread."""
        self.running = False
        self.write_queue.put(None)  # Shutdown signal
        self.thread.join(timeout=2.0)

class FrameWriterCallback(app_callback_class):
    """Custom callback class that writes frames to a file."""
    def __init__(self, output_path, write_all=False, copy_all=False):
        super().__init__()
        self.output_path = output_path
        self.write_all = write_all
        self.copy_all = copy_all
        self.frame_count = 0
        self.write_count = 0
        self.write_failures = 0

        # Initialize async writer
        self.async_writer = AsyncFrameWriter(max_queue_size=30)

        # Parse output path for numbered filenames
        output_path_obj = Path(output_path)
        self.output_dir = output_path_obj.parent
        self.output_stem = output_path_obj.stem
        self.output_ext = output_path_obj.suffix

        if self.write_all or self.copy_all:
            self.file_number = 1  # Start numbering from 1

    def write_frame(self, frame_rgb, frame_count, detection_count, detections, capture_timestamp):
        """Queue frame for asynchronous processing and writing (truly non-blocking)."""
        try:
            # Determine output path and rename info
            if self.write_all:
                # Write-all mode: Generate numbered filename for direct write
                output_file = self.output_dir / f"{self.output_stem}_{self.file_number:04d}{self.output_ext}"
                rename_from = None  # No rename needed
                self.file_number += 1
            elif self.copy_all:
                # Copy-all mode: Always write to same file, but rename previous
                output_file = self.output_path
                if self.file_number > 1:  # Only rename if there's a previous frame
                    rename_from = self.output_path
                    rename_to = self.output_dir / f"{self.output_stem}_{self.file_number-1:04d}{self.output_ext}"
                else:
                    rename_from = None
                    rename_to = None
                self.file_number += 1
                # Pass rename info to background thread
                rename_info = (rename_from, rename_to) if rename_from else None
            else:
                # Overwrite mode - use same file, no rename
                output_file = self.output_path
                rename_info = None

            # Queue frame for async processing and writing (instant, non-blocking)
            # All copying, text drawing, bounding box drawing, color conversion happens in background thread
            write_info = {
                'output_file': output_file,
                'rename_info': rename_info if self.copy_all else None,
                'frame_rgb': frame_rgb,
                'frame_count': frame_count,
                'detection_count': detection_count,
                'detections': detections,
                'capture_timestamp': capture_timestamp
            }

            if self.async_writer.write_async_with_rename(write_info):
                self.write_count += 1
                return True
            else:
                # Queue full - frame dropped
                self.write_failures += 1
                return False
        except Exception as e:
            print(f"Error queueing frame: {e}")
            self.write_failures += 1
            return False

    def shutdown(self):
        """Shutdown async writer and wait for pending writes."""
        self.async_writer.shutdown()

class FaceDetectionWithFileOutput(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        """
        Initialize the face detection app with frame file output.

        Args:
            app_callback: Callback function for processing frames
            user_data: User data object passed to callback
            parser: Argument parser (will be created if None)
        """
        setproctitle.setproctitle("Hailo Face Detection App")

        # Add our custom arguments before calling parent init
        if parser is None:
            parser = get_default_parser()
        parser.add_argument(
            "--output", "-o",
            type=str,
            default="/tmp/latest_face.jpg",
            help="Output file path (default: /tmp/latest_face.jpg)"
        )
        parser.add_argument(
            "--no-display",
            action="store_true",
            help="Disable video display window"
        )
        parser.add_argument(
            "--write-all",
            action="store_true",
            help="Write every frame to a new numbered file instead of overwriting"
        )
        parser.add_argument(
            "--copy-all",
            action="store_true",
            help="Rename existing output file to numbered backup before writing new frame"
        )

        # Call parent init which will parse args
        super().__init__(parser, user_data)

        # Detect architecture (hailo8 or hailo8l)
        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError("Could not auto-detect Hailo architecture. Please specify --arch manually.")
            self.arch = detected_arch
        else:
            self.arch = self.options_menu.arch

        # Set model paths based on architecture
        self.hef_path = get_resource_path(pipeline_name=FACE_DETECTION_PIPELINE, resource_type=RESOURCES_MODELS_DIR_NAME)
        self.post_process_so = get_resource_path(pipeline_name=None, resource_type=RESOURCES_SO_DIR_NAME, model=FACE_DETECTION_POSTPROCESS_SO_FILENAME)
        self.labels_json = get_resource_path(pipeline_name=None, resource_type=RESOURCES_JSON_DIR_NAME, model=FACE_DETECTION_JSON_NAME)

        # Choose detection function based on architecture
        if self.arch == "hailo8":
            self.detection_func = "scrfd_10g_letterbox"  # Higher accuracy model for Hailo-8
        else:  # hailo8l
            self.detection_func = "scrfd_2_5g_letterbox"  # Optimized model for Hailo-8L

        self.batch_size = 2

        print(f"Using architecture: {self.arch}")
        print(f"Using detection function: {self.detection_func}")
        print(f"HEF path: {self.hef_path}")
        print(f"Post-process SO: {self.post_process_so}")

        # Create the GStreamer pipeline
        self.create_pipeline()

        # Connect the callback to the pipeline
        self.app_callback = app_callback
        user_callback = self.pipeline.get_by_name("hailo_user_callback")
        if user_callback:
            user_callback_pad = user_callback.get_static_pad("src")
            user_callback_pad.add_probe(Gst.PadProbeType.BUFFER, self.app_callback, user_data)

    def get_pipeline_string(self):
        """
        Create the GStreamer pipeline for face detection with file output.
        """
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
            post_function_name=self.detection_func,
            batch_size=self.batch_size,
            config_json=self.labels_json
        )

        detection_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(detection_pipeline)

        # Optional tracker for face tracking
        tracker_pipeline = TRACKER_PIPELINE(class_id=-1, kalman_dist_thr=0.7, iou_thr=0.8)

        user_callback_pipeline = USER_CALLBACK_PIPELINE()

        # Use OVERLAY_PIPELINE to draw detections
        overlay_pipeline = OVERLAY_PIPELINE()

        # Use display or fakesink based on --no-display flag
        if self.options_menu.no_display:
            display_pipeline = f'{QUEUE(name="sink_queue")} ! fakesink sync={self.sync}'
        else:
            display_pipeline = DISPLAY_PIPELINE(
                video_sink=self.video_sink,
                sync=self.sync,
                show_fps=self.show_fps
            )

        pipeline_string = (
            f'{source_pipeline} ! '
            f'{detection_pipeline_wrapper} ! '
            f'{tracker_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{overlay_pipeline} ! '
            f'{display_pipeline}'
        )

        if self.options_menu.write_all:
            output_path_obj = Path(self.options_menu.output)
            print(f"Writing frames to: {output_path_obj.parent} (numbered files)")
        elif self.options_menu.copy_all:
            print(f"Writing frames to: {self.options_menu.output} (copy-all mode)")
        else:
            print(f"Writing frames to: {self.options_menu.output} (overwrite)")
        print(f"Display window: {'disabled' if self.options_menu.no_display else 'enabled'}")
        print(pipeline_string)
        return pipeline_string

# -----------------------------------------------------------------------------------------------
# Callback function for frame writing
# -----------------------------------------------------------------------------------------------
def face_detection_callback(pad, info, user_data):
    """
    Callback function that captures and queues frames for processing.

    Fast operations in callback (~2-3ms):
    - Copy frame data from GStreamer buffer
    - Extract face detection metadata (labels, bboxes, confidence)
    - Queue for background processing

    Slow operations in background thread (10-50ms):
    - Draw bounding boxes with labels
    - Add text overlays
    - Convert RGB to BGR
    - Write to disk
    """
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Increment frame counter
    user_data.increment()
    user_data.frame_count = user_data.get_count()

    # Get frame dimensions
    format, width, height = get_caps_from_pad(pad)

    # Validate frame format
    if not (format and width and height):
        if user_data.frame_count == 1:  # Only print on first frame to avoid spam
            print(f"Warning: Invalid frame format at frame {user_data.frame_count}")
        return Gst.PadProbeReturn.OK

    # Get frame data (this is a VIEW into buffer memory, very fast)
    frame = get_numpy_from_buffer(buffer, format, width, height)
    if frame is None:
        print(f"Error: Failed to extract frame {user_data.frame_count}")
        return Gst.PadProbeReturn.OK

    # CRITICAL: Must copy frame data BEFORE returning from callback
    # GStreamer will reuse the buffer memory once we return, invalidating the view
    # This copy is fast (~1-2ms) and necessary for correctness
    # Record capture timestamp for latency measurement
    capture_timestamp = time.time()
    frame_copy = frame.copy()

    # Get detections from buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Extract face detection data for background thread processing
    # Store as simple data structures (not Hailo objects which can't be passed to threads)
    detection_list = []
    detection_count = 0

    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()

        # Only process face detections
        if label == "face":
            # Store detection info as simple dict
            detection_info = (
                label,
                confidence,
                {
                    'xmin': bbox.xmin(),
                    'ymin': bbox.ymin(),
                    'width': bbox.width(),
                    'height': bbox.height()
                }
            )
            detection_list.append(detection_info)
            detection_count += 1

    # Queue frame copy for async processing and writing
    # Bounding box drawing, text overlays, color conversion, and disk I/O happen in background thread
    if user_data.write_frame(frame_copy, user_data.frame_count, detection_count, detection_list, capture_timestamp):
        if user_data.write_count % 30 == 0:  # Print every 30 frames
            if user_data.write_all:
                current_file = f"{user_data.output_stem}_{user_data.file_number-1:04d}{user_data.output_ext}"
                print(f"Frame {user_data.frame_count}: Queued {current_file} ({detection_count} faces)")
            elif user_data.copy_all:
                print(f"Frame {user_data.frame_count}: Queued to {user_data.output_path} (copy-all mode, {detection_count} faces)")
            else:
                print(f"Frame {user_data.frame_count}: Queued to {user_data.output_path} ({detection_count} faces)")

    # Return immediately - buffer flows through naturally
    # Pipeline continues at full speed
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str

    # Create user data with frame writer - will be initialized after arg parsing
    temp_user_data = FrameWriterCallback("/tmp/latest_face.jpg")

    # Create app with file output (arguments will be parsed inside)
    app = FaceDetectionWithFileOutput(face_detection_callback, temp_user_data)

    # Update user data with actual output path and flags from parsed args
    temp_user_data.output_path = app.options_menu.output
    temp_user_data.write_all = app.options_menu.write_all
    temp_user_data.copy_all = app.options_menu.copy_all

    # Re-initialize paths if write_all or copy_all is enabled
    if temp_user_data.write_all or temp_user_data.copy_all:
        output_path_obj = Path(temp_user_data.output_path)
        temp_user_data.output_dir = output_path_obj.parent
        temp_user_data.output_stem = output_path_obj.stem
        temp_user_data.output_ext = output_path_obj.suffix
        temp_user_data.file_number = 1

    # Validate output path after initialization
    output_path = Path(app.options_menu.output)
    output_dir = output_path.parent

    if not output_dir.exists():
        print(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    if not os.access(output_dir, os.W_OK):
        print(f"Error: No write permission for directory: {output_dir}")
        exit(1)

    # Get input source for display
    input_source = app.options_menu.input if app.options_menu.input else 'default video'

    print("\n" + "="*70)
    print("Face Detection with Frame File Output (SCRFD)")
    print("="*70)
    print(f"Input source: {input_source}")
    print(f"Display: {'Disabled' if app.options_menu.no_display else 'Enabled'}")

    if app.options_menu.write_all:
        print(f"Write mode: All frames (numbered)")
        print(f"Output directory: {output_dir}")
        print(f"Output pattern: {temp_user_data.output_stem}_####" + temp_user_data.output_ext)
    elif app.options_menu.copy_all:
        print(f"Write mode: Copy-all (rename previous before write)")
        print(f"Output file: {app.options_menu.output} (always latest)")
        print(f"Backup directory: {output_dir}")
        print(f"Backup pattern: {temp_user_data.output_stem}_####" + temp_user_data.output_ext)
    else:
        print(f"Write mode: Overwrite single file")
        print(f"Output file: {app.options_menu.output}")
        if output_path.exists():
            print(f"Note: Existing file will be overwritten")

    print("Press Ctrl+C to stop...")
    print("="*70 + "\n")

    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        # Shutdown async writer and wait for pending writes
        print("\nShutting down writer thread...")
        temp_user_data.shutdown()

        print("\n" + "="*70)
        print("Session Complete!")
        print("="*70)
        print(f"Total frames processed: {temp_user_data.frame_count}")
        print(f"Total frames written: {temp_user_data.write_count}")
        print(f"Total write failures (queue full): {temp_user_data.write_failures}")
        print(f"Frames dropped by async writer: {temp_user_data.async_writer.frames_dropped}")

        # Latency statistics
        if temp_user_data.async_writer.latency_count > 0:
            avg_latency = temp_user_data.async_writer.total_latency / temp_user_data.async_writer.latency_count
            print("\n" + "-"*70)
            print("Latency Statistics (Capture to Disk Write Complete)")
            print("-"*70)
            print(f"Average latency: {avg_latency*1000:.2f} ms")
            print(f"Min latency:     {temp_user_data.async_writer.min_latency*1000:.2f} ms")
            print(f"Max latency:     {temp_user_data.async_writer.max_latency*1000:.2f} ms")

            # Calculate percentiles if we have enough data
            if len(temp_user_data.async_writer.latencies) >= 10:
                sorted_latencies = sorted(temp_user_data.async_writer.latencies)
                p50 = sorted_latencies[len(sorted_latencies) // 2]
                p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
                p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
                print(f"P50 (median):    {p50*1000:.2f} ms")
                print(f"P95 latency:     {p95*1000:.2f} ms")
                print(f"P99 latency:     {p99*1000:.2f} ms")

        if app.options_menu.write_all:
            print(f"Output directory: {output_dir}")
            print(f"Files written: {temp_user_data.output_stem}_0001{temp_user_data.output_ext} to {temp_user_data.output_stem}_{temp_user_data.file_number-1:04d}{temp_user_data.output_ext}")

            # Calculate total size of all files
            total_size = 0
            for i in range(1, temp_user_data.file_number):
                file_path = output_dir / f"{temp_user_data.output_stem}_{i:04d}{temp_user_data.output_ext}"
                if file_path.exists():
                    total_size += file_path.stat().st_size
            total_size_mb = total_size / (1024 * 1024)
            print(f"Total size: {total_size_mb:.2f} MB")
        elif app.options_menu.copy_all:
            print(f"Latest frame saved to: {app.options_menu.output}")
            print(f"Backup directory: {output_dir}")
            if temp_user_data.file_number > 1:
                print(f"Backup files: {temp_user_data.output_stem}_0001{temp_user_data.output_ext} to {temp_user_data.output_stem}_{temp_user_data.file_number-2:04d}{temp_user_data.output_ext}")

                # Calculate total size of all files (including latest)
                total_size = 0
                # Add latest file
                if output_path.exists():
                    total_size += output_path.stat().st_size
                # Add all backup files
                for i in range(1, temp_user_data.file_number-1):
                    file_path = output_dir / f"{temp_user_data.output_stem}_{i:04d}{temp_user_data.output_ext}"
                    if file_path.exists():
                        total_size += file_path.stat().st_size
                total_size_mb = total_size / (1024 * 1024)
                print(f"Total size: {total_size_mb:.2f} MB")
            else:
                print("No backup files created")
        else:
            print(f"Latest frame saved to: {app.options_menu.output}")
            # Check if file exists and show info
            if output_path.exists():
                file_size = output_path.stat().st_size
                file_size_kb = file_size / 1024
                print(f"File size: {file_size_kb:.2f} KB")
            else:
                print("Warning: Output file was not created")

        print("="*70)
