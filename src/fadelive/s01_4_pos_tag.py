#!/usr/bin/env python3
"""
POS-Tagging für ein JSON-Vokabular. Nutzbar als CLI, als Python-Modul
oder als importierbare Pipeline-Funktion.

HINWEIS v3:
- Keine Änderungen nötig: Dieses Modul arbeitet nur mit JSON-Vokabularen,
  nicht direkt mit Metadaten aus dem Korpus
- Konsistent mit Pipeline v3

JSON-Format:

{
  "variant": "stop",
  "vocabulary_size": 170362,
  "top_words": [
    ["nicht", 45825],
    ["deutsch", 13701],
    ["ganz", 12999]
  ]
}

Beispielaufruf: 

    python s01_4_pos_tag.py \\
        --input output/vocabular/vocab_full_stop.json \\
        --output output/vocabular/vocab_top5000_stop_pos.csv \\
        --model de_core_news_lg
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Token


# ---------------------------------------------------------
# JSON LADEN
# ---------------------------------------------------------
def load_vocab(path, limit: int = 5000) -> List[Tuple[str, int]]:
    """
    Lädt ein JSON-Vokabular und gibt eine Liste mit den Top-N-Ausdrücken aus (word, count) zurück.
    Akzeptiert sowohl String- als auch Path-Argumente.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"JSON-Datei nicht gefunden: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "top_words" not in data:
        raise KeyError("Im JSON fehlt der Schlüssel 'top_words'.")

    vocab_raw = data["top_words"]
    vocab: List[Tuple[str, int]] = []

    for entry in vocab_raw:
        if isinstance(entry, list) and len(entry) == 2:
            word, cnt = entry
            if isinstance(word, str) and isinstance(cnt, int):
                vocab.append((word, cnt))

    if not vocab:
        raise ValueError("Keine gültigen Wort-Zähler-Paare in 'top_words' gefunden.")

    vocab_sorted = sorted(vocab, key=lambda x: x[1], reverse=True)
    vocab_limited = vocab_sorted[:limit]

    return vocab_limited


# ---------------------------------------------------------
# POS-TAGGING
# ---------------------------------------------------------

def get_pos(word: str, nlp: Language) -> str:
    """Gibt das POS-Tag eines Wortes zurück."""
    doc = nlp(word)
    for token in doc:
        if not token.is_space:
            return token.pos_
    return ""


def pos_tag_vocab(vocab: List[Tuple[str, int]], nlp: Language) -> pd.DataFrame:
    """Taggt alle Wörter und erstellt ein DataFrame."""
    rows = []
    for word, count in vocab:
        rows.append(
            {
                "word": word,
                "pos": get_pos(word, nlp),
                "count": count
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# PIPELINE-FUNKTION (für Python-Import)
# ---------------------------------------------------------
def run(
    input_json,
    output_csv="vocab_pos.csv",
    model="de_core_news_lg",
    limit=5000
):
    """
    Führt POS-Tagging auf einem JSON-Vokabular aus.
    
    Args:
        input_json: Pfad zur JSON-Vokabular-Datei
        output_csv: Pfad zur Ausgabe-CSV
        model: spaCy-Modellname
        limit: Anzahl der Top-Wörter
    
    Returns:
        Pfad zur erstellten CSV-Datei
    """
    vocab = load_vocab(input_json, limit=limit)

    nlp = spacy.load(model)

    df = pos_tag_vocab(vocab, nlp)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    
    print(f"✅ POS-Tagging abgeschlossen: {output_csv}")
    
    return output_csv


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="POS-Tagging eines JSON-Vokabulars")
    parser.add_argument("--input", required=True, help="Pfad zur JSON-Datei")
    parser.add_argument("--output", default="vocab_stop_pos.csv", help="Ausgabe-CSV")
    parser.add_argument("--model", default="de_core_news_lg", help="spaCy-Modellname")
    parser.add_argument("--limit", default=5000, type=int, help="Anzahl Top-Wörter")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    print(f"📥 Lade JSON: {args.input}")
    print(f"🧠 Lade Modell: {args.model}")

    output = run(args.input, args.output, args.model, args.limit)

    print(f"💾 Fertig! Ergebnis gespeichert unter: {output}")


if __name__ == "__main__":
    main()
