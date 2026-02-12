#!/usr/bin/env python3
"""
ai_summarizer.py

Summarize text using an AI model (OpenAI API).

"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from openai import OpenAI


def read_input(args: argparse.Namespace) -> str:
    if args.text:
        return args.text

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read()

    if args.stdin:
        return sys.stdin.read()

    print("Paste text to summarize, then end input (Ctrl+D on mac/linux, Ctrl+Z then Enter on windows):")
    return sys.stdin.read()


def build_prompt(style: str, max_words: int) -> str:
    if style == "bullets":
        style_instr = "Return a bullet-point summary (5–10 bullets)."
    elif style == "tl;dr":
        style_instr = "Return a TL;DR (1–2 sentences), then a short paragraph summary."
    else:
        style_instr = "Return a concise paragraph summary."

    return (
        f"{style_instr}\n"
        f"Keep it under about {max_words} words.\n"
        "Preserve key facts, numbers, names, and causal relationships.\n"
        "If the text is unclear or missing info, say so briefly.\n"
    )


def summarize(
    client: OpenAI,
    text: str,
    model: str,
    style: str,
    max_words: int,
) -> str:
    system_msg = "You are a helpful assistant that summarizes text accurately and concisely."
    prompt = build_prompt(style=style, max_words=max_words)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"{prompt}\n\nTEXT:\n{text}"},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize text using an AI model.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", help="Text to summarize.")
    group.add_argument("--file", help="Path to a UTF-8 text file to summarize.")
    group.add_argument("--stdin", action="store_true", help="Read text from stdin.")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model name to use.")
    parser.add_argument("--style", choices=["paragraph", "bullets", "tl;dr"], default="paragraph")
    parser.add_argument("--max-words", type=int, default=150, help="Target maximum word count.")
    args = parser.parse_args(argv)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set in your environment.", file=sys.stderr)
        return 2

    text = read_input(args).strip()
    if not text:
        print("ERROR: No input text provided.", file=sys.stderr)
        return 2

    client = OpenAI(api_key=api_key)

    try:
        summary = summarize(client=client, text=text, model=args.model, style=args.style, max_words=args.max_words)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
