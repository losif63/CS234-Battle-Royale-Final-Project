"""
Highlight Replay — Loads saved highlights and plays through the Renderer.

Controls (interactive mode):
    Space       = pause / unpause
    Left/Right  = step frame when paused
    R           = restart from beginning
    Escape      = quit

Usage:
    uv run python scripts/replay_highlight.py highlights/ace/game_001/
    uv run python scripts/replay_highlight.py highlights/ace/game_001/ --overview
    uv run python scripts/replay_highlight.py highlights/ace/game_001/ --mp4 ace.mp4
    uv run python scripts/replay_highlight.py highlights/ace/game_001/ --mp4 ace.mp4 --overview --speed 2
"""

import argparse
import json
import os
import pickle
import sys

import pygame

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from battle_royale.renderer import Renderer
from battle_royale.config import FPS, ARENA_W, ARENA_H, ENTITY_FOV_RADIUS


def load_highlight(game_dir: str) -> tuple[dict, list[dict]]:
    """Load metadata and frames from a highlight directory."""
    meta_path = os.path.join(game_dir, "metadata.json")
    frames_path = os.path.join(game_dir, "frames.pkl")

    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found")
        sys.exit(1)
    if not os.path.exists(frames_path):
        print(f"Error: {frames_path} not found")
        sys.exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)

    with open(frames_path, "rb") as f:
        frames = pickle.load(f)

    return metadata, frames


def _star_agent(metadata: dict) -> int | None:
    """Infer the 'star' agent from highlight metadata."""
    for key in ("winner", "agent", "killer"):
        if key in metadata:
            return int(metadata[key])
    return None


def _render_to_ffmpeg(game_dir: str, output_path: str, speed: float,
                      follow: int | None, resolution: int | None):
    """Render frames and stream directly to ffmpeg (no RAM buffering)."""
    import subprocess

    metadata, frames = load_highlight(game_dir)
    if follow == -1:
        follow = _star_agent(metadata)

    print(f"Category: {metadata.get('category', '?')}")
    print(f"Score: {metadata.get('score', '?')}")
    print(f"Frames: {len(frames)}")
    if follow is not None:
        print(f"Following: agent {follow}")

    if not frames:
        print("No frames to record.")
        return

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    renderer = Renderer(instant_restart=True, camera_follow=follow)
    info_font = pygame.font.SysFont("monospace", 14)

    # Determine native render size
    if follow is not None:
        native_w = native_h = int(ENTITY_FOV_RADIUS * 2)
    else:
        native_w, native_h = ARENA_W, ARENA_H

    # Output resolution: scale to fit within `resolution` if specified
    if resolution:
        scale = resolution / max(native_w, native_h)
        out_w = int(native_w * scale) // 2 * 2  # ensure even
        out_h = int(native_h * scale) // 2 * 2
    else:
        out_w = native_w // 2 * 2
        out_h = native_h // 2 * 2

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}", "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", "fast",
        output_path,
    ]
    print(f"Streaming to ffmpeg: {out_w}x{out_h} @ {FPS}fps -> {output_path}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Need a capture surface at output resolution
    capture_surf = pygame.Surface((out_w, out_h))

    frame_idx = 0
    frame_accum = 0.0
    written = 0

    try:
        while frame_idx < len(frames):
            state = frames[frame_idx]
            renderer.render(state)

            # HUD
            cat = metadata.get("category", "")
            hud_text = f"[{cat}] Frame {frame_idx+1}/{len(frames)}"
            hud_surf = info_font.render(hud_text, True, (255, 255, 100))
            renderer.screen.blit(hud_surf, (10, 10))

            # Scale to output resolution
            pygame.transform.scale(renderer.screen, (out_w, out_h), capture_surf)
            proc.stdin.write(pygame.image.tobytes(capture_surf, "RGB"))
            written += 1

            frame_accum += speed
            while frame_accum >= 1.0:
                frame_accum -= 1.0
                frame_idx += 1
                if frame_idx >= len(frames):
                    break

            if written % 500 == 0:
                print(f"  {written} frames written...")
    finally:
        proc.stdin.close()
        proc.wait()

    pygame.quit()

    if proc.returncode != 0:
        print(f"ffmpeg error: {proc.stderr.read().decode()}")
    else:
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"Saved: {output_path} ({size_mb:.1f} MB, {written} frames)")


def record_mp4(game_dir: str, output_path: str, speed: float = 1.0,
               follow: int | None = -1, resolution: int | None = None):
    """Record highlight to mp4, streaming frames to ffmpeg (constant RAM)."""
    _render_to_ffmpeg(game_dir, output_path, speed, follow, resolution)


def replay(game_dir: str, speed: float = 1.0, follow: int | None = -1):
    """Replay a saved highlight interactively."""
    metadata, frames = load_highlight(game_dir)

    if follow == -1:
        follow = _star_agent(metadata)

    print(f"Category: {metadata.get('category', '?')}")
    print(f"Score: {metadata.get('score', '?')}")
    print(f"Frames: {len(frames)}")
    if follow is not None:
        print(f"Following: agent {follow}")
    for k, v in metadata.items():
        if k not in ("category", "score", "num_frames"):
            print(f"  {k}: {v}")

    if not frames:
        print("No frames to replay.")
        return

    pygame.init()
    renderer = Renderer(instant_restart=True, camera_follow=follow)
    info_font = pygame.font.SysFont("monospace", 14)

    frame_idx = 0
    paused = False
    running = True
    clock = pygame.time.Clock()
    frame_accum = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    frame_idx = 0
                    frame_accum = 0.0
                    renderer.win_pause_counter = 0
                elif event.key == pygame.K_RIGHT and paused:
                    frame_idx = min(frame_idx + 1, len(frames) - 1)
                elif event.key == pygame.K_LEFT and paused:
                    frame_idx = max(frame_idx - 1, 0)

        if not running:
            break

        state = frames[frame_idx]
        renderer.render(state)

        screen = renderer.screen
        cat = metadata.get("category", "")
        status = "PAUSED" if paused else f"{speed}x"
        hud_text = f"[{cat}] Frame {frame_idx+1}/{len(frames)} | {status}"
        hud_surf = info_font.render(hud_text, True, (255, 255, 100))
        screen.blit(hud_surf, (10, 10))

        controls_text = "Space=pause  Arrows=step  R=restart  Esc=quit"
        ctrl_surf = info_font.render(controls_text, True, (180, 180, 180))
        sw = screen.get_width()
        screen.blit(ctrl_surf, (sw - ctrl_surf.get_width() - 10, 10))

        pygame.display.flip()

        if not paused:
            frame_accum += speed
            while frame_accum >= 1.0:
                frame_accum -= 1.0
                frame_idx += 1
                if frame_idx >= len(frames):
                    frame_idx = len(frames) - 1
                    paused = True
                    break

        clock.tick(FPS)

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Replay a saved highlight")
    parser.add_argument("game_dir", type=str,
                        help="Path to highlight game directory")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--follow", type=int, default=-1,
                        help="Agent to follow (-1=auto, use --overview for full arena)")
    parser.add_argument("--overview", action="store_true",
                        help="Full arena view (no follow)")
    parser.add_argument("--mp4", type=str, default=None,
                        help="Output path for mp4 recording (streams to ffmpeg, constant RAM)")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Max output dimension in pixels for mp4 (e.g. 1080, 1920)")
    args = parser.parse_args()

    follow = None if args.overview else args.follow

    if args.mp4:
        record_mp4(args.game_dir, args.mp4, speed=args.speed, follow=follow,
                   resolution=args.resolution)
    else:
        replay(args.game_dir, speed=args.speed, follow=follow)


if __name__ == "__main__":
    main()
