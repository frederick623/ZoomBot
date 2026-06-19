#!/usr/bin/env python3
"""
Zoom Meeting Bot - Automated Screenshot & Transcription
Joins Zoom meetings, captures slides on scene changes, and transcribes audio
"""

import cv2
import numpy as np
import whisper
import threading
import queue
import json
from datetime import datetime
from pathlib import Path
import pyaudio
import ffmpeg
import time
import logging
import platform
import torch
from tqdm import tqdm
import re
import subprocess
import webbrowser
from urllib.parse import urlparse, parse_qs, urlencode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_meeting_info(meeting_input: str, password: str = "") -> tuple[str, str]:
    """
    Extract (meeting_id, password) from a Zoom meeting URL or bare meeting ID.

    Supports:
    - https://zoom.us/j/<meeting_id>?<query_params>
    - https://<account>.zoom.us/j/<meeting_id>
    - Bare numeric IDs such as "123 456 7890" or "12345678901"

    Returns:
        (meeting_id, password) – password may be empty.
    """
    meeting_input = meeting_input.strip()
    pwd = password.strip()

    # Try to parse as a URL (add scheme if missing so urlparse works).
    url_str = meeting_input if meeting_input.startswith("http") else f"https://{meeting_input}"
    parsed = urlparse(url_str)
    if parsed.hostname and "zoom.us" in parsed.hostname:
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) >= 2 and path_parts[0] == "j":
            meeting_id = path_parts[1].strip()
            if not pwd:
                qs = parse_qs(parsed.query)
                pwd = qs.get("pwd", [""])[0]
            return meeting_id, pwd

    # Fall back: strip non-digits and treat the result as a bare meeting ID.
    digits = re.sub(r"\D", "", meeting_input)
    if 9 <= len(digits) <= 11:
        return digits, pwd

    raise ValueError(
        f"Could not extract a meeting ID from: {meeting_input!r}. "
        "Provide a full Zoom URL or a 9–11 digit meeting ID."
    )


def build_zoom_url(meeting_id: str, password: str = "") -> str:
    """Return a zoommtg:// deep-link URL for opening Zoom directly."""
    params = {"action": "join", "confno": meeting_id}
    if password:
        params["pwd"] = password
    return f"zoommtg://zoom.us/join?{urlencode(params)}"


class SceneDetector:
    """Detects scene changes in video frames"""
    
    def __init__(self, threshold: float = 0.25, min_interval: float = 2.0):
        """
        Args:
            threshold: Difference threshold to trigger scene change (0-1)
            min_interval: Minimum seconds between captures to avoid duplicates
        """
        self.threshold = threshold
        self.min_interval = min_interval
        self.last_frame = None
        self.last_capture_time = 0
        
    def detect_change(self, frame: np.ndarray) -> bool:
        """
        Check if current frame represents a scene change
        
        Args:
            frame: Current video frame (BGR format)
            
        Returns:
            True if scene change detected
        """
        current_time = time.time()
        
        # Respect minimum interval
        if current_time - self.last_capture_time < self.min_interval:
            return False
        
        # Convert to grayscale for comparison
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 240))  # Resize for faster processing
        
        if self.last_frame is None:
            self.last_frame = gray
            return True  # First frame is always a "change"
        
        # Compute structural similarity
        diff = cv2.absdiff(self.last_frame, gray)
        score = np.sum(diff) / (diff.shape[0] * diff.shape[1] * 255.0)
        
        if score > self.threshold:
            self.last_frame = gray
            self.last_capture_time = current_time
            logger.info(f"Scene change detected (score: {score:.3f})")
            return True
        
        return False


class AudioRecorder:
    """Records audio and handles Whisper transcription"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.current_chunk = []
        self.chunk_duration = 30  # seconds per chunk
        self.sample_rate = 16000
        self.channels = 1
        self.whisper_model = None
        
    def load_whisper(self, model_size: str = "turbo"):
        """Load Whisper model for transcription"""
        logger.info(f"Loading Whisper {model_size} model...")
        if platform.system() == "Darwin":
            self.whisper_model = whisper.load_model(model_size, "mps")
        elif not torch.cuda.is_available():
            self.whisper_model = whisper.load_model(model_size, "cpu")
        else:
            self.whisper_model = whisper.load_model(model_size)
        logger.info("Whisper model loaded")
        
    def start_recording(self):
        """Start audio recording in background thread"""
        self.is_recording = True
        threading.Thread(target=self._record_audio, daemon=True).start()
        threading.Thread(target=self._process_transcription, daemon=True).start()
        
    def stop_recording(self):
        """Stop audio recording"""
        self.is_recording = False
        
    def _record_audio(self):
        """Background thread for audio recording"""
        audio = pyaudio.PyAudio()
        
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        
        logger.info("Audio recording started")
        chunk_start = time.time()
        
        try:
            while self.is_recording:
                data = stream.read(1024, exception_on_overflow=False)
                self.current_chunk.append(data)
                
                # Save chunk every 30 seconds
                if time.time() - chunk_start > self.chunk_duration:
                    self._save_audio_chunk()
                    chunk_start = time.time()
                    
        except Exception as e:
            logger.error(f"Audio recording error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            # Save final chunk
            if self.current_chunk:
                self._save_audio_chunk()
                
    def _save_audio_chunk(self):
        if not self.current_chunk:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"audio_{timestamp}.mp3"
        raw_audio_data = b''.join(self.current_chunk)
        
        try:
            # Use ffmpeg-python's fluent interface
            process = (
                ffmpeg
                .input('pipe:0', format='s16le', acodec='pcm_s16le', ac=self.channels, ar=self.sample_rate)
                .output(str(filename), audio_bitrate='64k')
                .overwrite_output()
                .run_async(pipe_stdin=True, quiet=True)
            )
            
            # Pass the raw bytes to the process
            process.communicate(input=raw_audio_data)
            
            logger.info(f"Saved: {filename}")
            self.audio_queue.put(filename)
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
        finally:
            self.current_chunk = []
        
    def _process_transcription(self):
        """Background thread for Whisper transcription"""
        while self.is_recording or not self.audio_queue.empty():
            try:
                audio_file = self.audio_queue.get(timeout=1)
                self._transcribe_file(audio_file)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                
    def _transcribe_file(self, audio_file: Path):
        """Transcribe audio file with Whisper"""
        if not self.whisper_model:
            logger.warning("Whisper model not loaded, skipping transcription")
            return

        logger.info(f"Transcribing {audio_file.name}...")
        result = self.whisper_model.transcribe(str(audio_file), fp16=False, temperature=0.0)

        # Save transcript with separator annotations
        transcript_file = audio_file.with_suffix('.txt')
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(self._format_transcript(result))

        # Save detailed JSON with timestamps
        json_file = audio_file.with_suffix('.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Transcription saved: {transcript_file}")

    @staticmethod
    def _format_transcript(result: dict) -> str:
        """Format Whisper result into a readable, segmented transcript with separators"""
        lines = []
        for segment in result.get("segments", []):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            if not text:
                continue
            start_h = int(start // 3600)
            start_m = int((start % 3600) // 60)
            start_s = int(start % 60)
            end_h = int(end // 3600)
            end_m = int((end % 3600) // 60)
            end_s = int(end % 60)
            # lines.append(f"[{start_h:02d}:{start_m:02d}:{start_s:02d} - {end_h:02d}:{end_m:02d}:{end_s:02d}]")
            lines.append(text)
            # lines.append("-" * 40)
            # lines.append("")
        return "\n".join(lines)


class ZoomBot:
    """Main Zoom bot controller"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.screenshots_dir = self.output_dir / "screenshots"
        self.audio_dir = self.output_dir / "audio"
        self.screenshots_dir.mkdir(exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
        
        self.scene_detector = SceneDetector()
        self.audio_recorder = AudioRecorder(self.audio_dir)
        self.screenshot_count = 0
        
    def join_meeting(self, meeting_input: str, password: str = "", bot_name: str = "Recording Bot"):
        """
        Open a Zoom meeting in the installed Zoom client, then start the
        recording session (audio capture + Whisper transcription).

        The meeting is launched via the ``zoommtg://`` URL scheme so that the
        desktop Zoom app handles authentication and entry.  Once Zoom is open
        the existing audio/video capture pipeline begins automatically.

        Args:
            meeting_input: Zoom meeting URL (https://zoom.us/j/…) or bare
                           meeting ID (9–11 digits, spaces allowed).
            password:      Meeting passcode (optional; also extracted from URL).
            bot_name:      Display name shown in the meeting (unused when
                           launching via URL scheme – set in Zoom preferences).
        """
        meeting_id, pwd = parse_meeting_info(meeting_input, password)
        zoom_url = build_zoom_url(meeting_id, pwd)

        logger.info(f"Opening Zoom meeting {meeting_id} …")
        logger.info(f"Launch URL: {zoom_url}")

        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.Popen(["open", zoom_url])
            elif system == "Windows":
                subprocess.Popen(["start", zoom_url], shell=True)
            else:
                webbrowser.open(zoom_url)
        except Exception as exc:
            logger.warning(f"Could not open Zoom URL automatically: {exc}")
            logger.info(f"Please open this URL manually: {zoom_url}")

        # Give the Zoom client a moment to launch and join the meeting.
        logger.info("Waiting for Zoom to launch (10 s) …")
        time.sleep(10)

        logger.info("Starting recording session …")
        self.start_session()
        
    def process_video_frame(self, frame: np.ndarray):
        """
        Process incoming video frame
        
        Args:
            frame: Video frame in BGR format
        """
        if self.scene_detector.detect_change(frame):
            self._capture_screenshot(frame)
            
    def _capture_screenshot(self, frame: np.ndarray):
        """Save screenshot of current frame"""
        self.screenshot_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.screenshots_dir / f"slide_{self.screenshot_count:04d}_{timestamp}.png"
        
        cv2.imwrite(str(filename), frame)
        logger.info(f"Screenshot saved: {filename}")
        
    def start_session(self):
        """Start recording session"""
        logger.info("Starting recording session...")
        self.audio_recorder.load_whisper()
        self.audio_recorder.start_recording()
        
    def stop_session(self):
        """Stop recording and finalize"""
        logger.info("Stopping recording session...")
        self.audio_recorder.stop_recording()
        self._generate_summary()
        
    def _generate_summary(self):
        """Generate session summary"""
        summary_file = self.output_dir / "session_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"Zoom Meeting Recording Summary\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Screenshots captured: {self.screenshot_count}\n")
            f.write(f"Output directory: {self.output_dir.absolute()}\n\n")
            
            f.write("Files:\n")
            f.write(f"  Screenshots: {self.screenshots_dir}/\n")
            f.write(f"  Audio/Transcripts: {self.audio_dir}/\n")
            
        logger.info(f"Summary saved: {summary_file}")


# Demo/Testing functionality (simulates meeting with video file)
class DemoBot(ZoomBot):
    """Demo version that works with local video files for testing"""
    
    def process_video_file(self, video_path: str):
        """
        Process a video file as if it were a Zoom meeting
        Useful for testing without joining actual meetings
        
        Args:
            video_path: Path to video file
        """
        logger.info(f"Processing video file: {video_path}")
        
        self.start_session()
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        try:
            with tqdm(total=total_frames, desc="Processing video", unit="frames") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    pbar.update(1)

                    # Process every 10th frame to simulate real-time
                    if frame_count % 10 == 0:
                        self.process_video_frame(frame)

                    # Simulate real-time playback
                    time.sleep(0.033)  # ~30 fps

        finally:
            cap.release()
            self.stop_session()

        logger.info(f"Processed {frame_count} frames")

    def process_audio_file(self, audio_path: str):
        """
        Process an audio file (e.g., m4a) for transcription
        Useful for transcribing meeting recordings without video processing

        Args:
            audio_path: Path to audio file
        """
        logger.info(f"Processing audio file: {audio_path}")

        self.audio_recorder.load_whisper()

        audio_path = Path(audio_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with tqdm(total=1, desc="Transcribing audio") as pbar:
            result = self.audio_recorder.whisper_model.transcribe(str(audio_path))
            pbar.update(1)

        # Save transcript with separator annotations
        transcript_file = self.audio_dir / f"transcript_{timestamp}_{audio_path.stem}.txt"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(self.audio_recorder._format_transcript(result))

        # Save detailed JSON with timestamps
        json_file = self.audio_dir / f"transcript_{timestamp}_{audio_path.stem}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Transcription saved: {transcript_file}")
        logger.info(f"JSON saved: {json_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Zoom Meeting Bot")
    parser.add_argument("-v", "--video", help="Process video file (screenshots + transcription)")
    parser.add_argument("-a", "--audio", help="Process audio file (m4a transcription only)")
    parser.add_argument("-z", "--zoom", help="Zoom meeting URL or ID to join")
    parser.add_argument("-p", "--password", default="", help="Zoom meeting password/passcode")
    parser.add_argument("--output", default="output", help="Output directory")

    args = parser.parse_args()

    if args.video:
        bot = DemoBot(output_dir=args.output)
        bot.process_video_file(args.video)
    elif args.audio:
        bot = DemoBot(output_dir=args.output)
        bot.process_audio_file(args.audio)
    elif args.zoom:
        bot = ZoomBot(output_dir=args.output)
        try:
            bot.join_meeting(args.zoom, args.password)
        except ValueError as e:
            logger.error(str(e))
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
    else:
        parser.print_help()
