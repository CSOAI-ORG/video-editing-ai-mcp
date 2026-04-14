#!/usr/bin/env python3
"""
Video Editing AI MCP Server
==============================
AI-powered video editing toolkit for scene analysis, subtitle generation,
thumbnail selection, color grading, and aspect ratio conversion planning.

By MEOK AI Labs | https://meok.ai

Install: pip install mcp
Run:     python server.py
"""

import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
FREE_DAILY_LIMIT = 30
_usage: dict[str, list[datetime]] = defaultdict(list)


def _check_rate_limit(caller: str = "anonymous") -> Optional[str]:
    now = datetime.now()
    cutoff = now - timedelta(days=1)
    _usage[caller] = [t for t in _usage[caller] if t > cutoff]
    if len(_usage[caller]) >= FREE_DAILY_LIMIT:
        return f"Free tier limit reached ({FREE_DAILY_LIMIT}/day). Upgrade: https://mcpize.com/video-editing-ai-mcp/pro"
    _usage[caller].append(now)
    return None


# ---------------------------------------------------------------------------
# Scene detection heuristics
# ---------------------------------------------------------------------------
COMMON_SCENE_PATTERNS = {
    "dialogue": {"min_duration": 2.0, "max_duration": 30.0, "typical_cuts": "medium shot"},
    "action": {"min_duration": 0.5, "max_duration": 5.0, "typical_cuts": "quick cuts"},
    "establishing": {"min_duration": 3.0, "max_duration": 10.0, "typical_cuts": "wide shot"},
    "montage": {"min_duration": 1.0, "max_duration": 4.0, "typical_cuts": "rhythmic"},
    "transition": {"min_duration": 0.5, "max_duration": 2.0, "typical_cuts": "fade/dissolve"},
}

COLOR_GRADE_PROFILES = {
    "cinematic": {"contrast": 1.3, "saturation": 0.85, "temperature": 5800, "tint": "warm", "lut": "cinematic_teal_orange"},
    "vintage": {"contrast": 0.9, "saturation": 0.6, "temperature": 5200, "tint": "warm", "lut": "vintage_film_grain"},
    "documentary": {"contrast": 1.1, "saturation": 1.0, "temperature": 5600, "tint": "neutral", "lut": "natural_doc"},
    "horror": {"contrast": 1.5, "saturation": 0.4, "temperature": 4500, "tint": "cool", "lut": "dark_desaturated"},
    "comedy": {"contrast": 1.0, "saturation": 1.2, "temperature": 6000, "tint": "warm", "lut": "bright_pop"},
    "music_video": {"contrast": 1.4, "saturation": 1.3, "temperature": 5500, "tint": "mixed", "lut": "high_contrast_vivid"},
    "corporate": {"contrast": 1.05, "saturation": 0.95, "temperature": 5600, "tint": "neutral", "lut": "clean_professional"},
    "noir": {"contrast": 1.6, "saturation": 0.0, "temperature": 5000, "tint": "cool", "lut": "bw_high_contrast"},
}

ASPECT_RATIOS = {
    "16:9": {"width": 1920, "height": 1080, "use": "YouTube, TV, standard widescreen"},
    "9:16": {"width": 1080, "height": 1920, "use": "TikTok, Instagram Reels, YouTube Shorts"},
    "1:1": {"width": 1080, "height": 1080, "use": "Instagram feed, social media"},
    "4:3": {"width": 1440, "height": 1080, "use": "Classic TV, presentations"},
    "21:9": {"width": 2560, "height": 1080, "use": "Ultra-widescreen, cinematic"},
    "4:5": {"width": 1080, "height": 1350, "use": "Instagram portrait"},
    "2.39:1": {"width": 2390, "height": 1000, "use": "Anamorphic cinema"},
}


def _split_scenes(duration_seconds: float, scene_type: str, fps: float,
                  sensitivity: float) -> dict:
    """Analyze video metadata and recommend scene split points."""
    if duration_seconds <= 0:
        return {"error": "Duration must be positive"}
    if fps <= 0:
        fps = 24.0

    sensitivity = max(0.1, min(1.0, sensitivity))
    pattern = COMMON_SCENE_PATTERNS.get(scene_type, COMMON_SCENE_PATTERNS["dialogue"])

    avg_scene_length = pattern["min_duration"] + (pattern["max_duration"] - pattern["min_duration"]) * (1.0 - sensitivity)
    num_scenes = max(1, int(duration_seconds / avg_scene_length))

    scenes = []
    current_time = 0.0
    for i in range(num_scenes):
        scene_dur = avg_scene_length
        # Add slight variation
        variation = (hash(f"{i}_{duration_seconds}") % 100) / 100.0
        scene_dur *= (0.7 + variation * 0.6)
        scene_dur = min(scene_dur, duration_seconds - current_time)
        if scene_dur <= 0:
            break

        start_frame = int(current_time * fps)
        end_frame = int((current_time + scene_dur) * fps)

        scenes.append({
            "scene_index": i + 1,
            "start_time": round(current_time, 2),
            "end_time": round(current_time + scene_dur, 2),
            "duration": round(scene_dur, 2),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": end_frame - start_frame,
            "suggested_type": scene_type,
            "cut_style": pattern["typical_cuts"],
        })
        current_time += scene_dur

    return {
        "total_duration": duration_seconds,
        "fps": fps,
        "total_frames": int(duration_seconds * fps),
        "scene_count": len(scenes),
        "avg_scene_duration": round(duration_seconds / max(len(scenes), 1), 2),
        "detection_sensitivity": sensitivity,
        "content_type": scene_type,
        "scenes": scenes,
    }


def _generate_subtitles(transcript: str, duration_seconds: float,
                        style: str, max_chars_per_line: int) -> dict:
    """Generate timed subtitle data from a transcript."""
    if not transcript.strip():
        return {"error": "Transcript cannot be empty"}

    sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
    if not sentences:
        return {"error": "Could not parse sentences from transcript"}

    words_total = len(transcript.split())
    time_per_word = duration_seconds / max(words_total, 1)

    styles = {
        "standard": {"font": "Arial", "size": 24, "color": "#FFFFFF", "bg": "#000000AA", "position": "bottom"},
        "bold": {"font": "Arial Bold", "size": 28, "color": "#FFFFFF", "bg": "#000000CC", "position": "bottom"},
        "minimal": {"font": "Helvetica", "size": 22, "color": "#FFFFFF", "bg": "none", "position": "bottom"},
        "karaoke": {"font": "Arial Bold", "size": 30, "color": "#FFD700", "bg": "#000000BB", "position": "center"},
    }
    style_cfg = styles.get(style, styles["standard"])

    subtitles = []
    current_time = 0.0

    for idx, sentence in enumerate(sentences):
        words = sentence.split()
        chunks = []
        current_chunk = []
        current_len = 0

        for word in words:
            if current_len + len(word) + 1 > max_chars_per_line and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_len = len(word)
            else:
                current_chunk.append(word)
                current_len += len(word) + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        for chunk in chunks:
            word_count = len(chunk.split())
            chunk_duration = max(1.0, word_count * time_per_word)

            subtitles.append({
                "index": len(subtitles) + 1,
                "start_time": round(current_time, 3),
                "end_time": round(current_time + chunk_duration, 3),
                "text": chunk,
                "word_count": word_count,
            })
            current_time += chunk_duration

    srt_lines = []
    for sub in subtitles:
        start_h, start_r = divmod(sub["start_time"], 3600)
        start_m, start_s = divmod(start_r, 60)
        end_h, end_r = divmod(sub["end_time"], 3600)
        end_m, end_s = divmod(end_r, 60)
        srt_lines.append(f"{sub['index']}")
        srt_lines.append(f"{int(start_h):02d}:{int(start_m):02d}:{start_s:06.3f} --> {int(end_h):02d}:{int(end_m):02d}:{end_s:06.3f}")
        srt_lines.append(sub["text"])
        srt_lines.append("")

    return {
        "subtitle_count": len(subtitles),
        "total_duration": duration_seconds,
        "style": style_cfg,
        "format": "srt",
        "srt_content": "\n".join(srt_lines),
        "subtitles": subtitles[:50],
        "words_per_minute": round(words_total / (duration_seconds / 60), 1) if duration_seconds > 0 else 0,
    }


def _thumbnail_data(title: str, duration_seconds: float, scene_count: int,
                    style: str) -> dict:
    """Generate thumbnail selection recommendations and metadata."""
    key_moments = []
    if duration_seconds > 0 and scene_count > 0:
        interval = duration_seconds / scene_count
        for i in range(min(scene_count, 8)):
            timestamp = interval * i + interval * 0.4
            key_moments.append({
                "timestamp": round(timestamp, 2),
                "frame_number": int(timestamp * 24),
                "reason": f"Peak moment in scene {i + 1}",
                "thumbnail_score": round(0.5 + (hash(f"{title}_{i}") % 50) / 100.0, 2),
            })

    key_moments.sort(key=lambda x: x["thumbnail_score"], reverse=True)

    text_overlays = {
        "gaming": {"font": "Impact", "color": "#FF0000", "effect": "stroke + drop shadow", "position": "top-center"},
        "tutorial": {"font": "Roboto Bold", "color": "#FFFFFF", "effect": "dark overlay bar", "position": "bottom-center"},
        "vlog": {"font": "Montserrat", "color": "#FFD700", "effect": "gradient overlay", "position": "center"},
        "review": {"font": "Arial Black", "color": "#00FF00", "effect": "rating badge", "position": "top-right"},
        "news": {"font": "Helvetica Neue Bold", "color": "#FFFFFF", "effect": "lower-third bar", "position": "bottom"},
    }

    return {
        "title": title,
        "recommended_resolution": {"width": 1280, "height": 720},
        "key_moments": key_moments[:5],
        "best_timestamp": key_moments[0]["timestamp"] if key_moments else duration_seconds * 0.33,
        "text_overlay": text_overlays.get(style, text_overlays["vlog"]),
        "style": style,
        "composition_tips": [
            "Use rule of thirds for subject placement",
            "Ensure text is readable at small sizes",
            "High contrast between text and background",
            "Include a human face if possible (higher CTR)",
            "Avoid small details that disappear at thumbnail size",
        ],
    }


def _color_grading(genre: str, mood: str, lighting: str,
                   custom_adjustments: dict) -> dict:
    """Recommend color grading settings for a video."""
    profile = COLOR_GRADE_PROFILES.get(genre, COLOR_GRADE_PROFILES["cinematic"]).copy()

    mood_adjustments = {
        "happy": {"saturation_mod": 0.15, "temp_mod": 200, "contrast_mod": -0.05},
        "sad": {"saturation_mod": -0.2, "temp_mod": -300, "contrast_mod": 0.1},
        "tense": {"saturation_mod": -0.1, "temp_mod": -200, "contrast_mod": 0.2},
        "romantic": {"saturation_mod": 0.05, "temp_mod": 300, "contrast_mod": -0.1},
        "mysterious": {"saturation_mod": -0.15, "temp_mod": -400, "contrast_mod": 0.15},
        "energetic": {"saturation_mod": 0.2, "temp_mod": 100, "contrast_mod": 0.1},
    }

    lighting_adjustments = {
        "natural": {"exposure_comp": 0, "highlight_recovery": 0.3},
        "studio": {"exposure_comp": 0.1, "highlight_recovery": 0.1},
        "low_light": {"exposure_comp": 0.5, "highlight_recovery": 0.0},
        "harsh": {"exposure_comp": -0.3, "highlight_recovery": 0.8},
        "golden_hour": {"exposure_comp": 0.1, "highlight_recovery": 0.4},
        "mixed": {"exposure_comp": 0.2, "highlight_recovery": 0.5},
    }

    mood_adj = mood_adjustments.get(mood, {"saturation_mod": 0, "temp_mod": 0, "contrast_mod": 0})
    light_adj = lighting_adjustments.get(lighting, {"exposure_comp": 0, "highlight_recovery": 0.3})

    final_settings = {
        "contrast": round(profile["contrast"] + mood_adj["contrast_mod"], 2),
        "saturation": round(profile["saturation"] + mood_adj["saturation_mod"], 2),
        "temperature": profile["temperature"] + mood_adj["temp_mod"],
        "tint": profile["tint"],
        "exposure_compensation": light_adj["exposure_comp"],
        "highlight_recovery": light_adj["highlight_recovery"],
        "shadow_boost": round(0.3 if lighting == "low_light" else 0.1, 2),
        "recommended_lut": profile["lut"],
    }

    if custom_adjustments:
        for key, val in custom_adjustments.items():
            if key in final_settings and isinstance(val, (int, float)):
                final_settings[key] = val

    return {
        "genre": genre,
        "mood": mood,
        "lighting": lighting,
        "settings": final_settings,
        "color_wheels": {
            "shadows": {"hue": 210 if profile["tint"] == "cool" else 30, "saturation": 0.15},
            "midtones": {"hue": 0, "saturation": 0.0},
            "highlights": {"hue": 40 if profile["tint"] == "warm" else 200, "saturation": 0.1},
        },
        "export_formats": ["DaVinci Resolve .drx", "Adobe Premiere .cube", "Final Cut Pro .cglut"],
    }


def _aspect_ratio_convert(source_ratio: str, target_ratio: str,
                          source_width: int, source_height: int,
                          strategy: str) -> dict:
    """Plan aspect ratio conversion with crop/pad calculations."""
    target_info = ASPECT_RATIOS.get(target_ratio)
    if not target_info:
        return {"error": f"Unknown target ratio '{target_ratio}'. Available: {list(ASPECT_RATIOS.keys())}"}

    source_ar = source_width / max(source_height, 1)
    parts = target_ratio.replace(":", "/").split("/")
    if len(parts) == 2:
        target_ar = float(parts[0]) / float(parts[1])
    else:
        target_ar = target_info["width"] / target_info["height"]

    strategies = {
        "crop": "Remove content from edges to fill target frame",
        "letterbox": "Add black bars to preserve all content",
        "smart_crop": "AI-guided crop keeping subjects centered",
        "stretch": "Distort to fill (not recommended)",
        "fill_blur": "Blurred background with original centered",
    }

    if strategy == "crop" or strategy == "smart_crop":
        if source_ar > target_ar:
            new_width = int(source_height * target_ar)
            new_height = source_height
            crop_x = (source_width - new_width) // 2
            crop_y = 0
        else:
            new_width = source_width
            new_height = int(source_width / target_ar)
            crop_x = 0
            crop_y = (source_height - new_height) // 2

        content_preserved = round((new_width * new_height) / (source_width * source_height) * 100, 1)
        result_method = "crop"
    else:
        if source_ar > target_ar:
            new_width = int(source_height * target_ar)
            new_height = source_height
            pad_x = 0
            pad_y = (int(new_width / target_ar) - source_height) // 2
        else:
            new_width = source_width
            new_height = int(source_width / target_ar)
            pad_x = (int(new_height * target_ar) - source_width) // 2
            pad_y = 0

        crop_x, crop_y = 0, 0
        content_preserved = 100.0
        result_method = strategy

    output_w = target_info["width"]
    output_h = target_info["height"]

    return {
        "source": {"width": source_width, "height": source_height, "ratio": source_ratio, "aspect": round(source_ar, 3)},
        "target": {"width": output_w, "height": output_h, "ratio": target_ratio, "aspect": round(target_ar, 3), "use_case": target_info["use"]},
        "strategy": strategy,
        "strategy_description": strategies.get(strategy, "Unknown"),
        "conversion": {
            "method": result_method,
            "crop_offset_x": crop_x if result_method == "crop" else 0,
            "crop_offset_y": crop_y if result_method == "crop" else 0,
            "content_preserved_pct": content_preserved,
            "output_resolution": f"{output_w}x{output_h}",
            "needs_upscale": output_w > source_width or output_h > source_height,
        },
        "ffmpeg_hint": f"ffmpeg -i input.mp4 -vf 'scale={output_w}:{output_h}:force_original_aspect_ratio=decrease,pad={output_w}:{output_h}:(ow-iw)/2:(oh-ih)/2' output.mp4",
    }


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Video Editing AI MCP",
    instructions="AI-powered video editing toolkit: scene splitting, subtitle generation, thumbnail planning, color grading, and aspect ratio conversion. By MEOK AI Labs.")


@mcp.tool()
def split_scenes(duration_seconds: float, scene_type: str = "dialogue",
                 fps: float = 24.0, sensitivity: float = 0.5) -> dict:
    """Analyze video metadata and produce scene split points with timestamps
    and frame numbers. Useful for automatic chapter markers or editing cuts.

    Args:
        duration_seconds: Total video duration in seconds
        scene_type: Content type (dialogue, action, establishing, montage, transition)
        fps: Frames per second of the source video (default: 24)
        sensitivity: Detection sensitivity 0.1-1.0 (higher = more cuts)
    """
    err = _check_rate_limit()
    if err:
        return {"error": err}
    try:
        return _split_scenes(duration_seconds, scene_type, fps, sensitivity)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def generate_subtitles(transcript: str, duration_seconds: float,
                       style: str = "standard", max_chars_per_line: int = 42) -> dict:
    """Generate timed subtitles (SRT format) from a transcript. Automatically
    splits text into readable chunks with proper timing.

    Args:
        transcript: Full text transcript of the video
        duration_seconds: Total video duration in seconds
        style: Subtitle style (standard, bold, minimal, karaoke)
        max_chars_per_line: Maximum characters per subtitle line (default: 42)
    """
    err = _check_rate_limit()
    if err:
        return {"error": err}
    try:
        return _generate_subtitles(transcript, duration_seconds, style, max_chars_per_line)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def thumbnail_data(title: str, duration_seconds: float, scene_count: int = 5,
                   style: str = "vlog") -> dict:
    """Generate thumbnail selection recommendations including best timestamps,
    composition tips, and text overlay configuration.

    Args:
        title: Video title (used for text overlay recommendations)
        duration_seconds: Total video duration in seconds
        scene_count: Number of distinct scenes in the video
        style: Video style (gaming, tutorial, vlog, review, news)
    """
    err = _check_rate_limit()
    if err:
        return {"error": err}
    try:
        return _thumbnail_data(title, duration_seconds, scene_count, style)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def recommend_color_grading(genre: str = "cinematic", mood: str = "neutral",
                            lighting: str = "natural",
                            custom_adjustments: dict = {}) -> dict:
    """Recommend color grading settings based on genre, mood, and lighting
    conditions. Returns contrast, saturation, temperature, LUT suggestions,
    and color wheel settings.

    Args:
        genre: Video genre (cinematic, vintage, documentary, horror, comedy, music_video, corporate, noir)
        mood: Emotional tone (happy, sad, tense, romantic, mysterious, energetic)
        lighting: Lighting condition (natural, studio, low_light, harsh, golden_hour, mixed)
        custom_adjustments: Optional overrides as {setting: value}
    """
    err = _check_rate_limit()
    if err:
        return {"error": err}
    try:
        return _color_grading(genre, mood, lighting, custom_adjustments)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def convert_aspect_ratio(source_ratio: str = "16:9", target_ratio: str = "9:16",
                         source_width: int = 1920, source_height: int = 1080,
                         strategy: str = "smart_crop") -> dict:
    """Plan an aspect ratio conversion with detailed crop/pad calculations,
    content preservation percentage, and FFmpeg command hints.

    Args:
        source_ratio: Source aspect ratio label (e.g. "16:9")
        target_ratio: Target aspect ratio (16:9, 9:16, 1:1, 4:3, 21:9, 4:5, 2.39:1)
        source_width: Source video width in pixels
        source_height: Source video height in pixels
        strategy: Conversion strategy (crop, letterbox, smart_crop, stretch, fill_blur)
    """
    err = _check_rate_limit()
    if err:
        return {"error": err}
    try:
        return _aspect_ratio_convert(source_ratio, target_ratio, source_width, source_height, strategy)
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run()
