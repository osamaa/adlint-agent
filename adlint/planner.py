"""Planners: given what perception saw, pick the next tool call.

Both planners get the same context dict and return the same action dict:
    {"action": <name>, "args": {...}, "reason": <text>}
The rule planner always works. The LLM planner (OpenRouter) is optional; any
bad, slow or repeated answer from it falls back to the rule planner.
"""

import json
import logging
import os
import time

import requests

log = logging.getLogger("adlint.planner")

SOURCE_ACTIONS = ["enhance_exposure", "sharpen", "accept", "escalate"]
PLACEMENT_ACTIONS = ["resize", "smart_crop", "fit_into_safe_zone", "accept", "escalate"]

FIXABLE_BY_REFRAME = {
    "aspect_mismatch", "face_cut", "logo_cut", "text_cut", "subject_cut",
    "face_in_ui_zone", "text_in_ui_zone", "logo_in_ui_zone",
}
NEEDS_HUMAN_AT_SOURCE = {"severely_blurry", "overexposed", "text_heavy", "logo_missing"}


def _tried(context, action, strategy=None):
    for step in context["tried"]:
        if step["action"] == action and (strategy is None or step.get("args", {}).get("strategy") == strategy):
            return True
    return False


class RulePlanner:
    name = "rules"

    def decide(self, context):
        if context["stage"] == "source":
            return self._decide_source(context)
        return self._decide_placement(context)

    def _decide_source(self, context):
        codes = {issue["code"] for issue in context["issues"]}
        if codes & {"dark", "low_contrast"} and not _tried(context, "enhance_exposure"):
            return {"action": "enhance_exposure", "args": {}, "reason": "image is dark or flat: lift exposure, then re-check"}
        if "blurry" in codes and "severely_blurry" not in codes and not _tried(context, "sharpen"):
            return {"action": "sharpen", "args": {}, "reason": "mild softness: unsharp mask, then re-check"}
        blocking = sorted(codes & NEEDS_HUMAN_AT_SOURCE)
        leftover = sorted(codes - NEEDS_HUMAN_AT_SOURCE)  # fixes already tried and still failing
        if blocking or leftover:
            return {"action": "escalate", "args": {}, "reason": "cannot fix automatically: " + ", ".join(blocking + leftover)}
        return {"action": "accept", "args": {}, "reason": "source passes all quality checks"}

    def _decide_placement(self, context):
        codes = {issue["code"] for issue in context["issues"]}
        if not context["has_candidate"]:
            if context["aspect_match"]:
                return {"action": "resize", "args": {}, "reason": "aspect already matches: plain resize"}
            return {"action": "smart_crop", "args": {"strategy": "all"}, "reason": "aspect differs: crop around faces, logo, text and subject"}
        if not codes:
            return {"action": "accept", "args": {}, "reason": "candidate passes every placement check"}
        if codes & FIXABLE_BY_REFRAME:
            if not context["aspect_match"] and not _tried(context, "smart_crop", "all"):
                return {"action": "smart_crop", "args": {"strategy": "all"}, "reason": "try a crop that keeps everything"}
            if not context["aspect_match"] and context["priority_differs"] and not _tried(context, "smart_crop", "priority"):
                return {"action": "smart_crop", "args": {"strategy": "priority"},
                        "reason": "full crop failed: keep faces and logo first, let text/subject edges go"}
            if not _tried(context, "fit_into_safe_zone"):
                return {"action": "fit_into_safe_zone", "args": {},
                        "reason": "crops cannot keep it all in the safe zone: shrink into the safe zone on a blurred backdrop"}
        if codes & {"text_too_small", "logo_not_redetected"} and context["candidate_kind"] == "fit" \
                and not context["aspect_match"] and not _tried(context, "smart_crop", "priority"):
            return {"action": "smart_crop", "args": {"strategy": "priority"},
                    "reason": "shrinking made text/logo too small: try a tighter crop instead"}
        return {"action": "escalate", "args": {}, "reason": "no tool left that fixes: " + ", ".join(sorted(codes))}


LLM_SYSTEM = """You are the planning step of AdLint, an agent that prepares ad creatives for ad placements.
OpenCV has measured the creative; you choose the NEXT tool call. Reply with JSON only:
{"action": "<tool>", "args": {...}, "reason": "<one short sentence>"}

Tools for stage "source" (the master creative):
- enhance_exposure {}: gamma + CLAHE. Use for dark / low_contrast.
- sharpen {}: unsharp mask. Only for mild "blurry", never for "severely_blurry".
- accept {}: no issues left.
- escalate {}: hand to a human (unfixable: severely_blurry, overexposed, text_heavy, logo_missing, or a fix already tried that did not help).

Tools for stage "placement" (one output size):
- resize {}: when aspect_match is true and there is no candidate yet.
- smart_crop {"strategy": "all"|"priority"}: crop at the target aspect. "all" keeps faces+logo+text+subject; "priority" keeps faces+logo only.
- fit_into_safe_zone {}: shrink the whole creative into the safe zone with a blurred backdrop. Keeps everything, but makes text smaller.
- accept {}: the current candidate has no issues.
- escalate {}: nothing left that would fix the issues.

Rules: never repeat a tool+args already in "tried". Accept only when "issues" is empty. Prefer the least destructive tool."""


class LLMPlanner:
    """Asks an LLM (via OpenRouter) for the next step; rule planner is the safety net."""

    name = "llm"

    def __init__(self, api_key, model=None, timeout=30):
        self.api_key = api_key
        self.model = model or os.environ.get("ADLINT_LLM_MODEL", "deepseek/deepseek-v4.1-flash")
        self.timeout = timeout
        self.fallback = RulePlanner()

    def decide(self, context):
        try:
            action = self._ask(context)
        except Exception as error:  # network, JSON, schema: all fall back, and say so in the trace
            log.warning("llm planner failed, using rules: %s", error)
            action = None
            failure = str(error)[:200]
        else:
            failure = self._invalid(action, context)
        if failure is None:
            action["planner"] = f"llm:{self.model}"
            return action
        fallback = self.fallback.decide(context)
        fallback["planner"] = "rules (llm fallback)"
        fallback["llm_problem"] = failure
        return fallback

    def _ask(self, context):
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM},
                {"role": "user", "content": json.dumps(context)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": 600,
            "reasoning": {"effort": "low"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(2):
            response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                     json=body, headers=headers, timeout=self.timeout)
            if response.status_code in (429, 500, 502, 503) and attempt == 0:
                time.sleep(2)
                continue
            response.raise_for_status()
            break
        text = response.json()["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        return json.loads(text)

    @staticmethod
    def _invalid(action, context):
        if not isinstance(action, dict) or "action" not in action:
            return "answer is not an action object"
        allowed = SOURCE_ACTIONS if context["stage"] == "source" else PLACEMENT_ACTIONS
        if action["action"] not in allowed:
            return f"unknown tool {action['action']!r}"
        action.setdefault("args", {})
        if not isinstance(action["args"], dict):
            return "args is not an object"
        if action["action"] == "smart_crop" and action["args"].get("strategy") not in ("all", "priority"):
            action["args"]["strategy"] = "all"
        if action["action"] not in ("accept", "escalate") and any(
            step["action"] == action["action"] and step.get("args", {}) == action["args"] for step in context["tried"]
        ):
            return "repeats a tool call already tried"
        if action["action"] == "accept" and context["issues"]:
            return "accepted while issues remain"
        action.setdefault("reason", "")
        return None


def make_planner(kind="auto"):
    """kind: rules | llm | auto (llm when OPENROUTER_API_KEY is set)."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if kind == "llm" and not key:
        raise SystemExit("planner=llm needs OPENROUTER_API_KEY in the environment")
    if kind == "llm" or (kind == "auto" and key):
        return LLMPlanner(key)
    return RulePlanner()
