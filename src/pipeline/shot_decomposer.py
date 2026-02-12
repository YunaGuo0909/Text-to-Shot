"""
Shot Decomposer Module.

Takes a full scene description and decomposes it into a sequence of 
individual shot prompts using LLM-based script analysis. Each shot 
includes a text description, recommended shot type, and ordering.

This is a NEW module extending the original single-shot generation 
toward multi-shot storyboard generation.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ShotConfig:
    """Configuration for a single shot in the storyboard."""
    shot_index: int
    description: str
    shot_type: str  # close-up, medium-shot, wide-shot, over-the-shoulder, two-shot
    camera_motion: str = "static"  # static, dolly-in, dolly-out, pan-left, pan-right, crane-up, crane-down, track, orbit
    character_a_action: str = ""
    character_b_action: str = ""
    duration_hint: float = 3.0  # seconds
    notes: str = ""


@dataclass
class StoryboardPlan:
    """Complete storyboard plan from scene decomposition."""
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


class ShotDecomposer:
    """
    Decomposes a scene description into a sequence of cinematic shots.
    
    Uses LLM (OpenAI API or local model) to analyze narrative structure
    and produce shot-level breakdowns following cinematographic conventions.
    """

    # System prompt for LLM-based decomposition
    SYSTEM_PROMPT = """You are a professional film storyboard artist and cinematographer. 
Your task is to decompose a scene description into a sequence of cinematic shots.

For each shot, provide:
1. "shot_index": Sequential number starting from 1
2. "description": A concise description of what happens in this shot (for two characters A and B)
3. "shot_type": One of ["close-up", "medium-shot", "wide-shot", "over-the-shoulder", "two-shot"]
4. "camera_motion": One of ["static", "dolly-in", "dolly-out", "pan-left", "pan-right", "crane-up", "crane-down", "track", "orbit"]
5. "character_a_action": What character A is doing
6. "character_b_action": What character B is doing
7. "duration_hint": Estimated duration in seconds (1-8)
8. "notes": Any cinematographic notes (mood, lighting, etc.)

Follow these cinematographic principles:
- Start with an establishing wide shot to set the scene
- Use shot/reverse-shot for dialogue or confrontation
- Close-ups for emotional moments or important details
- Maintain the 180-degree rule across consecutive shots
- Vary shot types for visual interest
- Use over-the-shoulder shots for conversations
- Dolly-in for dramatic emphasis, dolly-out for reveals
- Tracking shots for following character movement
- Crane shots for establishing spatial context

Output ONLY a valid JSON array of shot objects."""

    def __init__(self, llm_provider="openai", model_name="gpt-4", api_key=None):
        """
        Args:
            llm_provider: LLM provider ("openai" or "local")
            model_name: Model name for the provider
            api_key: API key (for OpenAI)
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.api_key = api_key

    def decompose(self, scene_description: str, max_shots: int = 8) -> StoryboardPlan:
        """
        Decompose a scene description into shot sequence.
        
        Args:
            scene_description: Full text description of the scene
            max_shots: Maximum number of shots to generate
            
        Returns:
            StoryboardPlan with ordered shot configurations
        """
        user_prompt = f"""Decompose the following scene into {max_shots} or fewer cinematic shots:

Scene: "{scene_description}"

Remember to output ONLY a valid JSON array."""

        if self.llm_provider == "openai":
            shots_json = self._call_openai(user_prompt)
        else:
            shots_json = self._call_local(user_prompt)

        # Parse response
        shots = self._parse_shots(shots_json)
        
        return StoryboardPlan(
            scene_description=scene_description,
            shots=shots,
            total_shots=len(shots)
        )

    def decompose_manual(self, shot_descriptions: List[dict]) -> StoryboardPlan:
        """
        Create a storyboard plan from manually specified shots.
        Useful for testing without LLM dependency.
        
        Args:
            shot_descriptions: List of dicts with shot parameters
            
        Returns:
            StoryboardPlan
        """
        shots = []
        for i, desc in enumerate(shot_descriptions):
            shots.append(ShotConfig(
                shot_index=i + 1,
                description=desc.get("description", ""),
                shot_type=desc.get("shot_type", "medium-shot"),
                camera_motion=desc.get("camera_motion", "static"),
                character_a_action=desc.get("character_a_action", ""),
                character_b_action=desc.get("character_b_action", ""),
                duration_hint=desc.get("duration_hint", 3.0),
                notes=desc.get("notes", ""),
            ))
        return StoryboardPlan(
            scene_description="Manual storyboard",
            shots=shots,
            total_shots=len(shots)
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
                    {"role": "user", "content": user_prompt}
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
        # TODO: Implement local LLM inference (e.g., using transformers library)
        raise NotImplementedError("Local LLM inference not yet implemented")

    def _parse_shots(self, json_str: str) -> List[ShotConfig]:
        """Parse LLM response into ShotConfig list."""
        try:
            # Clean up response (remove markdown code blocks if present)
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
                    character_a_action=data.get("character_a_action", ""),
                    character_b_action=data.get("character_b_action", ""),
                    duration_hint=data.get("duration_hint", 3.0),
                    notes=data.get("notes", ""),
                ))
            return shots
        except json.JSONDecodeError as e:
            print(f"Failed to parse shot decomposition: {e}")
            return []
