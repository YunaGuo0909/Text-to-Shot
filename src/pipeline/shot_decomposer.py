"""
Shot Decomposer Module.

Takes a screenplay or scene description and decomposes it into a sequence
of individual shots using LLM-based script analysis. Each shot specifies
the shot type, camera motion, and narrative context — focused on camera
decisions rather than character poses.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ShotConfig:
    """Configuration for a single shot in the sequence."""
    shot_index: int
    description: str
    shot_type: str  # close-up, medium-shot, wide-shot, over-the-shoulder, two-shot
    camera_motion: str = "static"
    duration_hint: float = 3.0  # seconds
    emotional_tone: str = ""  # e.g., tense, calm, dramatic, playful
    notes: str = ""


@dataclass
class StoryboardPlan:
    """Complete shot plan from scene decomposition."""
    scene_description: str
    shots: List[ShotConfig] = field(default_factory=list)
    total_shots: int = 0


# Shot type mapping for model conditioning
SHOT_TYPE_MAP = {
    "close-up": 0,
    "medium-shot": 1,
    "wide-shot": 2,
    "over-the-shoulder": 3,
    "two-shot": 4,
}

# Camera motion type mapping
CAMERA_MOTION_MAP = {
    "static": 0,
    "dolly-in": 1,
    "dolly-out": 2,
    "pan-left": 3,
    "pan-right": 4,
    "crane-up": 5,
    "crane-down": 6,
    "track": 7,
    "orbit": 8,
}


class ShotDecomposer:
    """
    Decomposes a scene description into a sequence of cinematic shots.

    Uses LLM (OpenAI API or local model) to analyze narrative structure
    and produce shot-level breakdowns with camera motion decisions.
    """

    SYSTEM_PROMPT = """You are a professional cinematographer and camera operator.
Your task is to decompose a scene description into a sequence of cinematic shots,
focusing on CAMERA DECISIONS — shot type, camera movement, duration, and mood.

For each shot, provide:
1. "shot_index": Sequential number starting from 1
2. "description": What is happening in this shot (narrative context for camera decisions)
3. "shot_type": One of ["close-up", "medium-shot", "wide-shot", "over-the-shoulder", "two-shot"]
4. "camera_motion": One of ["static", "dolly-in", "dolly-out", "pan-left", "pan-right", "crane-up", "crane-down", "track", "orbit"]
5. "duration_hint": Estimated duration in seconds (1-8)
6. "emotional_tone": The mood this shot conveys (e.g., "tense", "calm", "dramatic", "intimate")
7. "notes": Cinematographic reasoning for your camera choices

Follow these cinematographic principles:
- Start with an establishing wide shot to set the scene
- Use shot/reverse-shot for dialogue or confrontation
- Close-ups for emotional moments or important details
- Maintain the 180-degree rule across consecutive shots
- Vary shot types for visual interest
- Dolly-in for dramatic emphasis, dolly-out for reveals
- Tracking shots for following character movement
- Crane shots for establishing spatial context
- Match camera motion to emotional beats

Output ONLY a valid JSON array of shot objects."""

    def __init__(self, llm_provider="openai", model_name="gpt-4", api_key=None):
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.api_key = api_key

    def decompose(self, scene_description: str, max_shots: int = 8) -> StoryboardPlan:
        """
        Decompose a scene description into a camera shot sequence.

        Args:
            scene_description: Full text description of the scene
            max_shots: Maximum number of shots

        Returns:
            StoryboardPlan with ordered shot configurations
        """
        user_prompt = f"""Decompose the following scene into {max_shots} or fewer cinematic shots.
Focus on camera decisions: what shot type and camera motion best serves each narrative beat.

Scene: "{scene_description}"

Output ONLY a valid JSON array."""

        if self.llm_provider == "openai":
            shots_json = self._call_openai(user_prompt)
        else:
            shots_json = self._call_local(user_prompt)

        shots = self._parse_shots(shots_json)

        return StoryboardPlan(
            scene_description=scene_description,
            shots=shots,
            total_shots=len(shots),
        )

    def decompose_manual(self, shot_descriptions: List[dict]) -> StoryboardPlan:
        """
        Create a shot plan from manually specified shots.
        Useful for testing without LLM dependency.
        """
        shots = []
        for i, desc in enumerate(shot_descriptions):
            shots.append(ShotConfig(
                shot_index=i + 1,
                description=desc.get("description", ""),
                shot_type=desc.get("shot_type", "medium-shot"),
                camera_motion=desc.get("camera_motion", "static"),
                duration_hint=desc.get("duration_hint", 3.0),
                emotional_tone=desc.get("emotional_tone", ""),
                notes=desc.get("notes", ""),
            ))
        return StoryboardPlan(
            scene_description="Manual shot plan",
            shots=shots,
            total_shots=len(shots),
        )

    def _call_openai(self, user_prompt: str) -> str:
        """Call OpenAI API for shot decomposition."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            return "[]"

    def _call_local(self, user_prompt: str) -> str:
        """Call local LLM for shot decomposition (placeholder)."""
        raise NotImplementedError("Local LLM inference not yet implemented")

    def _parse_shots(self, json_str: str) -> List[ShotConfig]:
        """Parse LLM response into ShotConfig list."""
        try:
            json_str = json_str.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]

            shots_data = json.loads(json_str)
            shots = []
            for data in shots_data:
                shots.append(ShotConfig(
                    shot_index=data.get("shot_index", len(shots) + 1),
                    description=data.get("description", ""),
                    shot_type=data.get("shot_type", "medium-shot"),
                    camera_motion=data.get("camera_motion", "static"),
                    duration_hint=data.get("duration_hint", 3.0),
                    emotional_tone=data.get("emotional_tone", ""),
                    notes=data.get("notes", ""),
                ))
            return shots
        except json.JSONDecodeError as e:
            print(f"Failed to parse shot decomposition: {e}")
            return []
