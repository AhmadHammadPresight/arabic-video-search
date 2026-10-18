"""FFmpeg helpers: probe a video, decode sampled frames, and write browser-playable copies.

Decoding asks for VideoToolbox (the Apple media engine); FFmpeg falls back to software
decoding by itself for codecs VideoToolbox does not handle, such as MPEG-1.
Encoding uses the VideoToolbox H.264 encoder, so no GPL encoder (libx264) is involved;
an upload that is already H.264 is remuxed instead, so its playback copy loses nothing.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

H264_ENCODER = os.environ.get("AVS_H264_ENCODER", "h264_videotoolbox")


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    pix_fmt: str = ""


def probe(path: Path) -> VideoInfo:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate:stream_side_data=rotation:format=duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise MediaError(f"ffprobe failed for {path.name}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise MediaError(f"no video stream in {path.name}")
    stream = streams[0]
    width, height = int(stream["width"]), int(stream["height"])
    rotation = 0
    for side in stream.get("side_data_list") or []:
        rotation = int(side.get("rotation", 0) or 0)
    if abs(rotation) % 180 == 90:
        # FFmpeg auto-rotates on decode, so frames come out with swapped sides.
        width, height = height, width
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate")) or 0.0
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    return VideoInfo(
        duration=duration, width=width, height=height, fps=fps,
        codec=stream.get("codec_name", ""), pix_fmt=stream.get("pix_fmt", ""),
    )


def _parse_rate(rate: str | None) -> float:
    if not rate or rate in ("0/0", "N/A"):
        return 0.0
    num, _, den = rate.partition("/")
    return float(num) / float(den or 1)


def iter_frames(path: Path, sample_fps: float, info: VideoInfo | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (seconds from start, RGB frame) at `sample_fps` frames per second."""
    info = info or probe(path)
    frame_bytes = info.width * info.height * 3
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-hwaccel", "videotoolbox",
        "-i", str(path), "-an", "-sn",
        "-vf", f"fps={sample_fps}", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    index = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(info.height, info.width, 3)
            yield index / sample_fps, frame
            index += 1
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        returncode = proc.wait()
    if returncode != 0 and index == 0:
        raise MediaError(f"ffmpeg could not decode {path.name}: {stderr.strip()}")


def read_frame(path: Path, t: float, info: VideoInfo | None = None) -> np.ndarray:
    """One RGB frame at `t` seconds, decoded the same way as `iter_frames`."""
    info = info or probe(path)
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-hwaccel", "videotoolbox", "-ss", f"{t:.3f}",
        "-i", str(path), "-an", "-sn", "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    frame_bytes = info.width * info.height * 3
    if result.returncode != 0 or len(result.stdout) < frame_bytes:
        raise MediaError(f"ffmpeg could not read the frame at {t:.2f}s of {path.name}: {result.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(result.stdout[:frame_bytes], dtype=np.uint8).reshape(info.height, info.width, 3)


def encode_h264(src: Path, dst: Path, start: float | None = None, duration: float | None = None) -> None:
    """Write a silent, browser-playable H.264 MP4 copy of `src`, optionally trimmed."""
    info = probe(src)
    # About 0.1 bit per pixel per frame: 6 Mbit/s for 1080p30, with a floor for tiny videos.
    bitrate = max(500_000, int(0.1 * info.width * info.height * (info.fps or 25)))
    cmd = ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", H264_ENCODER, "-b:v", str(bitrate), "-pix_fmt", "yuv420p"]
    _ffmpeg_to_mp4(src, dst, cmd, start, duration)


def make_playback_copy(src: Path, dst: Path) -> None:
    """Browser-playable copy of an upload: a lossless remux when it is already H.264 4:2:0, else a re-encode."""
    info = probe(src)
    if info.codec == "h264" and info.pix_fmt in ("yuv420p", "yuvj420p"):
        _ffmpeg_to_mp4(src, dst, ["-c:v", "copy"], input_args=["-fflags", "+genpts"])
    else:
        encode_h264(src, dst)


def _ffmpeg_to_mp4(
    src: Path,
    dst: Path,
    codec_args: list[str],
    start: float | None = None,
    duration: float | None = None,
    input_args: tuple[str, ...] | list[str] = (),
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".partial.mp4")
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *input_args]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1", *codec_args, "-movflags", "+faststart", str(tmp)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise MediaError(f"ffmpeg could not write {dst.name} from {src.name}: {result.stderr.strip()}")
    tmp.replace(dst)
