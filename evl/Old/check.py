import argparse
import os
import sys
import traceback

from google import genai
from google.genai import types

# Default target (you can override via --model or GEMINI_MODEL env)
TEST_MODEL = "gemini-3-pro-preview"

FALLBACK_ORDER = [
    "gemini-3-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro-preview-03-25",
    "gemini-pro-latest",
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
]


def list_models(client):
    print("Listing available models …")
    try:
        models = list(client.models.list())
        for m in models:
            try:
                actions = getattr(m, "supported_actions", None) or []
                print(f"- {m.name} | supports: {actions}")
            except Exception:
                print(f"- {m.name}")
        return models
    except Exception as e:
        print("Error calling list_models():", e)
        traceback.print_exc()
        return None


def pick_model(target_name: str, models):
    """Pick a model that supports generateContent. Prefer target; else fallbacks."""
    # Normalize names to a set for quick checks
    available = {m.name.replace("models/", ""): m for m in models}

    def supports_generate(name):
        m = available.get(name)
        if not m:
            return False
        actions = getattr(m, "supported_actions", []) or []
        return "generateContent" in actions

    # 1) exact target
    if target_name and supports_generate(target_name):
        return target_name

    # 2) try best match by prefix (e.g., gemini-3-pro-preview-* variants)
    if target_name:
        candidates = [
            n
            for n in available.keys()
            if n == target_name or n.startswith(target_name + "-")
        ]
        for c in candidates:
            if supports_generate(c):
                return c

    # 3) fallbacks in preferred order
    for name in FALLBACK_ORDER:
        # allow "-something" suffix variants too
        if name in available and supports_generate(name):
            return name
        for n in available.keys():
            if n == name or n.startswith(name + "-"):
                if supports_generate(n):
                    return n

    # 4) last resort: any model with generateContent
    for n, m in available.items():
        if "generateContent" in (getattr(m, "supported_actions", []) or []):
            return n

    return None


def test_model_call(client, model: str, prompt: str):
    print(f"\nTesting model call with model = '{model}' …")
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = getattr(resp, "text", None)
        if text is None:
            # Fallback: try candidates list
            candidates = getattr(resp, "candidates", [])
            if candidates and getattr(candidates[0], "content", None):
                parts = candidates[0].content.parts or []
                text = (
                    "\n".join(
                        [
                            getattr(p, "text", "")
                            for p in parts
                            if getattr(p, "text", "")
                        ]
                    )
                    or None
                )

        print("Model response:")
        print(text if text is not None else "(no text returned)")

        # Print served model info if present
        served = getattr(resp, "model", None)
        cand_model = None
        if getattr(resp, "candidates", None):
            cand_model = getattr(resp.candidates[0], "model", None)

        if served or cand_model:
            print("— served_model:", served)
            print("— candidate_model:", cand_model)

        return True
    except Exception as e:
        print("Error during generate_content():", e)
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Check Google GenAI API access and model availability."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("GEMINI_MODEL", TEST_MODEL),
        help="Model to test (default from GEMINI_MODEL env or gemini-3-pro-preview)",
    )
    parser.add_argument(
        "--api-version",
        type=str,
        default=os.getenv("GENAI_API_VERSION", "v1beta"),
        help="API version, e.g. v1beta or v1 (default v1beta)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Find the race condition in this multi-threaded C++ snippet: [code here]",
        help="Prompt to send for the test call.",
    )
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Please set GOOGLE_API_KEY or GEMINI_API_KEY.")
        sys.exit(1)

    # Build client
    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version=args.api_version),
        )
    except Exception as e:
        print("❌ Failed to create client:", e)
        traceback.print_exc()
        sys.exit(1)

    print("✅ Client created successfully.")
    models = list_models(client)
    if models is None:
        print("❌ Failed to list models — cannot proceed.")
        sys.exit(1)

    chosen = pick_model(args.model, models)
    if not chosen:
        print(
            f"❌ Could not find a model that supports generateContent (requested: {args.model})."
        )
        sys.exit(1)

    # Show chosen model’s supported actions
    chosen_meta = None
    for m in models:
        if (
            m.name.endswith(chosen)
            or m.name == chosen
            or m.name.replace("models/", "") == chosen
        ):
            chosen_meta = m
            break
    actions = getattr(chosen_meta, "supported_actions", []) if chosen_meta else []
    print(f"\nSelected model: {chosen}")
    print(f"Supported actions: {actions}")

    ok = test_model_call(client, chosen, args.prompt)
    if not ok:
        print(
            "❌ Model call failed — check model name, API key, permissions, or try a different --api-version."
        )
        sys.exit(1)

    print("\n🎉 API check finished successfully.")


if __name__ == "__main__":
    main()
