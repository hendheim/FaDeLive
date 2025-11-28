#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Erstellt Vokabulare aus den Preprocessing-Ausgaben:

- Vokabular des gesamten Korpus
- Vokabular je Textklasse
- Vokabular je Zeitintervall
- Vokabular je Genre

Input-Dateien:
    korpus_min.csv
    korpus_lem.csv
    korpus_stop.csv

Ausgabe:
    vocab_full_<variant>.json
    vocab_textclass_<variant>_<textclass>.json
    vocab_interval_<variant>_<interval>.json
    vocab_genre_<variant>_<genre>.json

Beispielaufruf: 

    python src/fadelive/s01_02_vocabular.py `
        --input-dir output/processed_corpus `
        --output-dir output/vocabular `
        --delimiter ";"

"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter
from typing import Iterable, Dict, List, Tuple

import pandas as pd


# ---------------------------------------------------------
# Hilfsfunktion: Tokenisierung & Frequenzen
# ---------------------------------------------------------

def analyze_vocabulary(texts: Iterable[str]) -> Counter:
    """Erstellt ein Frequenzvokabular aus einer Sequenz von Texten."""
    all_text = " ".join(t for t in texts if isinstance(t, str)).lower()
    tokens = re.findall(r"\b\w+\b", all_text)
    return Counter(tokens)


# ---------------------------------------------------------
# Speichern
# ---------------------------------------------------------

def save_vocab(data: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"> gespeichert: {out_path}")


# ---------------------------------------------------------
# Genres wie in deinem Original
# ---------------------------------------------------------

PREDEFINED_GENRES = [
    "Theorie", "Methodik", "Erläuterung", "Lexikonartikel",
    "Verordnung", "Rezension"
]

# ---------------------------------------------------------
# Zeitintervalle wie in deinem Original
# ---------------------------------------------------------

INTERVALS = {
    "1782-1852": (1782, 1852),
    "1853-1864": (1853, 1864),
    "1865-1876": (1865, 1876),
    "1877-1891": (1877, 1891),
    "1782-1856": (1782, 1856),
    "1857-1872": (1857, 1872),
    "1873-1891": (1873, 1891),
    "1782-1864": (1782, 1864),
    "1865-1891": (1865, 1891)
}

# ---------------------------------------------------------
# Vokabularerstellung für jede Variante
# ---------------------------------------------------------

def build_vocabularies(df: pd.DataFrame, variant: str, output_dir: Path):
    """
    Erstellt Vokabulare für:
        - Gesamt
        - Textklassen
        - Zeitintervalle
        - Genres
    """

    # -----------------------------------------------------
    # 1) Gesamtvokabular
    # -----------------------------------------------------
    print(f"\nErzeuge Gesamtvokabular ({variant}) …")

    freq = analyze_vocabulary(df["content"])
    vocab_data = {
        "variant": variant,
        "vocabulary_size": len(freq),
        "top_words": freq.most_common(5000),
        "full_vocab": dict(freq)
    }
    save_vocab(vocab_data, output_dir / f"vocab_full_{variant}.json")

    # -----------------------------------------------------
    # 2) Textklassen
    # -----------------------------------------------------
    print(f"Erzeuge Vokabulare für Textklassen ({variant}) …")
    if "textclass" in df.columns:
        for tc in sorted(df["textclass"].dropna().unique()):
            texts = df.loc[df["textclass"] == tc, "content"].astype(str).tolist()
            if not texts:
                continue

            freq = analyze_vocabulary(texts)
            vocab_data = {
                "variant": variant,
                "textclass": tc,
                "vocabulary_size": len(freq),
                "top_words": freq.most_common(5000),
                "full_vocab": dict(freq)
            }
            out = output_dir / "textclass" / f"vocab_textclass_{variant}_{tc}.json"
            save_vocab(vocab_data, out)

    # -----------------------------------------------------
    # 3) Zeitintervalle
    # -----------------------------------------------------
    print(f"Erzeuge Vokabulare für Zeitintervalle ({variant}) …")
    if "year" in df.columns:
        for label, (start_y, end_y) in INTERVALS.items():
            mask = df["year"].apply(lambda y: isinstance(y, (int, float)) and start_y <= y <= end_y)
            texts = df.loc[mask, "content"].astype(str).tolist()
            if not texts:
                continue

            freq = analyze_vocabulary(texts)
            vocab_data = {
                "variant": variant,
                "interval": label,
                "year_range": [start_y, end_y],
                "vocabulary_size": len(freq),
                "top_words": freq.most_common(5000),
                "full_vocab": dict(freq)
            }
            out = output_dir / "intervals" / f"vocab_interval_{variant}_{label}.json"
            save_vocab(vocab_data, out)

    # -----------------------------------------------------
    # 4) Genres
    # -----------------------------------------------------
    print(f"Erzeuge Vokabulare für Genres ({variant}) …")
    if "genre" in df.columns:
        for genre in PREDEFINED_GENRES:
            # prüfe, ob der Eintrag Teilstrings enthält (kommagetrennte Listen)
            mask = df["genre"].astype(str).str.contains(
                rf"(^|, )?{re.escape(genre)}($|, )?",
                regex=True, na=False
            )
            texts = df.loc[mask, "content"].astype(str).tolist()
            if not texts:
                continue

            freq = analyze_vocabulary(texts)
            vocab_data = {
                "variant": variant,
                "genre": genre,
                "vocabulary_size": len(freq),
                "top_words": freq.most_common(5000),
                "full_vocab": dict(freq)
            }
            out = output_dir / "genres" / f"vocab_genre_{variant}_{genre}.json"
            save_vocab(vocab_data, out)


# ---------------------------------------------------------
# Argumente
# ---------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Erstellt Vokabulare aus Preprocessing-Outputs.")
    p.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Ordner mit korpus_min.csv, korpus_lem.csv, korpus_stop.csv",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Ordner zum Speichern der Vokabulardateien (JSON).",
    )
    p.add_argument(
        "--delimiter",
        default="\t",
        help="CSV/TSV-Feldtrenner (Standard: Tab)",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------
# Run-Funktion (für Pipeline / direkten Funktionsaufruf)
# ---------------------------------------------------------

def run(
    input_dir: Path,
    output_dir: Path,
    delimiter: str = "\t",
) -> None:
    variants = ["min", "lem", "stop"]

    for variant in variants:
        infile = input_dir / f"korpus_{variant}.csv"
        if not infile.exists():
            print(f"(!) Datei fehlt: {infile} — überspringe.")
            continue

        print(f"\nLese {infile} …")
        df = pd.read_csv(infile, sep=delimiter, encoding="utf-8")

        # Es muss eine Spalte "content" geben
        if "content" not in df.columns:
            raise ValueError(f"Datei {infile} enthält keine Spalte 'content'.")

        build_vocabularies(df, variant, output_dir)

    print("\nFertig. Alle Vokabulare erstellt.")


# ---------------------------------------------------------
# Main (CLI-Wrapper)
# ---------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        delimiter=args.delimiter,
    )


if __name__ == "__main__":
    main()
