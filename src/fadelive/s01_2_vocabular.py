#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Erstellt Vokabulare aus den Preprocessing-Ausgaben.

ÄNDERUNG v2:
- Arbeitet mit den neuen Content-Spaltennamen: content_min, content_lem, content_stop
- Flexible Metadaten-Handhabung
- year/year_first werden speziell für Zeitintervalle verwendet
- textclass und genre werden dynamisch erkannt (falls vorhanden)

Input-Dateien:
    korpus_min.csv (mit content_min)
    korpus_lem.csv (mit content_lem)
    korpus_stop.csv (mit content_stop)

Ausgabe:
    vocab_full_<variant>.json
    vocab_textclass_<variant>_<textclass>.json (falls textclass-Spalte existiert)
    vocab_interval_<variant>_<interval>.json (falls year/year_first existiert)
    vocab_genre_<variant>_<genre>.json (falls genre-Spalte existiert)

Beispielaufruf: 

    python s01_2_vocabular_v2.py \
        --input-dir output/processed_corpus \
        --output-dir output/vocabular \
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
# Metadaten-Erkennung
# ---------------------------------------------------------

def identify_content_column(df: pd.DataFrame) -> str:
    """
    Identifiziert die Content-Spalte (content_min, content_lem oder content_stop).
    
    Returns:
        Name der Content-Spalte
    """
    for col in ["content_stop", "content_lem", "content_min"]:
        if col in df.columns:
            return col
    raise ValueError("Keine Content-Spalte gefunden (content_min/content_lem/content_stop)")


def has_column(df: pd.DataFrame, col: str) -> bool:
    """Prüft, ob eine Spalte existiert und nicht-leere Werte enthält."""
    return col in df.columns and df[col].notna().any()


# ---------------------------------------------------------
# Speichern
# ---------------------------------------------------------

def save_vocab(data: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✔ gespeichert: {out_path}")


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
        - Textklassen (falls Spalte vorhanden)
        - Zeitintervalle (falls year/year_first vorhanden)
        - Genres (falls genre-Spalte vorhanden)
    """
    
    # Content-Spalte identifizieren
    content_col = identify_content_column(df)
    print(f"  📋 Content-Spalte: {content_col}")

    # -----------------------------------------------------
    # 1) Gesamtvokabular
    # -----------------------------------------------------
    print(f"  🔄 Erzeuge Gesamtvokabular ({variant}) …")

    freq = analyze_vocabulary(df[content_col])
    vocab_data = {
        "variant": variant,
        "vocabulary_size": len(freq),
        "top_words": freq.most_common(5000),
        "full_vocab": dict(freq)
    }
    save_vocab(vocab_data, output_dir / f"vocab_full_{variant}.json")

    # -----------------------------------------------------
    # 2) Textklassen (falls vorhanden)
    # -----------------------------------------------------
    if has_column(df, "textclass"):
        print(f"  🔄 Erzeuge Vokabulare für Textklassen ({variant}) …")
        for tc in sorted(df["textclass"].dropna().unique()):
            texts = df.loc[df["textclass"] == tc, content_col].astype(str).tolist()
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
    else:
        print(f"  ⚠️  Keine 'textclass'-Spalte gefunden – Textklassen-Vokabulare übersprungen.")

    # -----------------------------------------------------
    # 3) Zeitintervalle (falls year/year_first vorhanden)
    # -----------------------------------------------------
    if has_column(df, "year") or has_column(df, "year_first"):
        print(f"  🔄 Erzeuge Vokabulare für Zeitintervalle ({variant}) …")
        
        # year_first hat Vorrang, sonst year
        if "year_first" in df.columns:
            year_col = df["year_first"].combine_first(df.get("year", pd.Series()))
        else:
            year_col = df["year"]
        
        year_col = pd.to_numeric(year_col, errors="coerce")
        
        for label, (start_y, end_y) in INTERVALS.items():
            mask = year_col.apply(lambda y: isinstance(y, (int, float)) and start_y <= y <= end_y)
            texts = df.loc[mask, content_col].astype(str).tolist()
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
    else:
        print(f"  ⚠️  Keine 'year' oder 'year_first'-Spalte gefunden – Intervall-Vokabulare übersprungen.")

    # -----------------------------------------------------
    # 4) Genres (falls vorhanden)
    # -----------------------------------------------------
    if has_column(df, "genre"):
        print(f"  🔄 Erzeuge Vokabulare für Genres ({variant}) …")
        for genre in PREDEFINED_GENRES:
            # prüfe, ob der Eintrag Teilstrings enthält (kommagetrennte Listen)
            mask = df["genre"].astype(str).str.contains(
                rf"(^|, )?{re.escape(genre)}($|, )?",
                regex=True, na=False
            )
            texts = df.loc[mask, content_col].astype(str).tolist()
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
    else:
        print(f"  ⚠️  Keine 'genre'-Spalte gefunden – Genre-Vokabulare übersprungen.")


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
            print(f"⚠️  Datei fehlt: {infile} – überspringe.")
            continue

        print(f"\n{'='*70}")
        print(f"VARIANTE: {variant.upper()}")
        print(f"{'='*70}")
        print(f"📄 Lese {infile} …")
        df = pd.read_csv(infile, sep=delimiter, encoding="utf-8")

        build_vocabularies(df, variant, output_dir)

    print("\n" + "="*70)
    print("✅ Fertig. Alle Vokabulare erstellt.")
    print("="*70)


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
