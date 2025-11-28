#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trainiert Word2Vec-Modelle auf Basis eines oder mehrerer vorverarbeiteter
Korpora 'korpus_gen*.csv' (Gensim-Preprocessing, stopwortbereinigt, lemmatisiert).

Erwartetes Eingabeformat:
    - CSV-Dateien mit mindestens der Spalte: "content"
    - Beispiel: output/processed_corpus/korpus_gen.csv
                output/processed_corpus/korpus_gen-Erl.csv

Für jede Eingabedatei wird:
    1) die Spalte "content" geladen,
    2) der Inhalt in Sätze und Tokens zerlegt (NLTK, Deutsch),
    3) ein Word2Vec-Modell mit festem Parameter-Set trainiert,
    4) das Modell als .model-Datei gespeichert.

Standard-Parameter:
    vector_size = 175
    window      = 10
    min_count   = 13  (Parameter-Variante 3)
    workers     = Anzahl CPU-Kerne
    epochs      = 20
    sg          = 1   (Skip-gram)
    hs          = 0   (Negative Sampling)
    negative    = 10
    sample      = 1e-4

Beispielaufruf:

    python src/fadelive/s07_word-vector-model.py `
        --input-dir output/processed_corpus `
        --pattern "korpus_gen*.csv" `
        --output-dir output/word2vec_models

        
"""

import argparse
import os
from pathlib import Path

import nltk
import pandas as pd
from gensim.models import Word2Vec

# ---------------------------------------------------------
# Parameter
# ---------------------------------------------------------

VECTOR_SIZE = 175
WINDOW = 10
MIN_COUNT = 13
WORKERS = 4
EPOCHS = 20
SG = 1
NEGATIVE = 10
SAMPLE = 1e-4

# ---------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------

def find_input_files(input_dir: Path, pattern: str) -> list[Path]:
    """Sucht alle Dateien im input_dir, die zum Pattern passen (kein rekursives Suchen)."""
    return sorted(input_dir.glob(pattern))


def tokenize_corpus(texts: pd.Series) -> list[list[str]]:
    """
    Nimmt eine Serie von Texten (Strings) und gibt eine Liste von Sätzen,
    wobei jeder Satz eine Tokenliste ist.
    """
    sentences = []

    for raw_text in texts.dropna():
        raw_text = str(raw_text).strip()
        if not raw_text:
            continue

        # In Sätze zerlegen (Deutsch)
        try:
            sents = nltk.sent_tokenize(raw_text, language="german")
        except LookupError as e:
            raise RuntimeError(
                "NLTK 'punkt' Tokenizer für Deutsch fehlt.\n"
                "Bitte einmalig ausführen:\n"
                "    import nltk\n"
                '    nltk.download("punkt")'
            ) from e

        # Jeden Satz tokenisieren
        for sent in sents:
            tokens = nltk.word_tokenize(sent, language="german")
            # Leersätze überspringen
            if tokens:
                sentences.append(tokens)

    return sentences


def train_word2vec(sentences) -> Word2Vec:
    """Trainiert ein Word2Vec-Modell auf Basis der gegebenen Sätze."""
    
    model = Word2Vec(
        sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        epochs=EPOCHS,
        sg=SG,
        negative=NEGATIVE,
        sample=SAMPLE,
    )
    return model


def process_file(input_file: Path, output_dir: Path):
    """Lädt eine CSV-Datei, tokenisiert die Texte und trainiert ein Word2Vec-Modell."""
    print(f"📄 Verarbeite Datei: {input_file}")

    if not input_file.exists():
        print(f"⚠️ Datei existiert nicht, übersprungen: {input_file}")
        return

    try:
        df = pd.read_csv(input_file, encoding="utf-8", sep=";")
    except Exception as e:
        print(f"⚠️ Fehler beim Einlesen von {input_file}: {e}")
        return

    if "content" not in df.columns:
        print(f"⚠️ Spalte 'content' fehlt in {input_file}, übersprungen.")
        return

    texts = df["content"].astype(str)
    if texts.str.strip().eq("").all():
        print(f"⚠️ Alle Einträge in 'content' sind leer in {input_file}, übersprungen.")
        return

    print("   ➜ Tokenisiere Korpus …")
    sentences = tokenize_corpus(texts)

    if not sentences:
        print(f"⚠️ Keine Sätze nach Tokenisierung in {input_file}, übersprungen.")
        return

    print(f"   ➜ Trainiere Word2Vec-Modell (Sätze: {len(sentences)}) …")
    try:
        model = train_word2vec(sentences)
    except Exception as e:
        print(f"❌ Fehler beim Training des Modells für {input_file}: {e}")
        return

    # Modell speichern
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"{input_file.stem}.model"
    model_path = output_dir / model_name

    model.save(str(model_path))
    print(f"   ✔ Modell gespeichert unter: {model_path}")


# ---------------------------------------------------------
# run-Funktion für Pipeline
# ---------------------------------------------------------

def run(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "korpus_gen*.csv",
) -> None:
    """Trainiert Word2Vec-Modelle auf Basis von korpus_gen-CSV-Dateien."""

    print(f"📁 Eingabeordner: {input_dir}")
    print(f"📁 Ausgabeordner: {output_dir}")
    print(f"🔎 Dateipattern:  {pattern}")

    input_files = find_input_files(input_dir, pattern)
    if not input_files:
        print("⚠️ Keine passenden Eingabedateien gefunden.")
        return

    print(f"✔ {len(input_files)} Datei(en) gefunden.\n")

    for f in input_files:
        process_file(f, output_dir)

    print("\n✅ Alle Modelle wurden verarbeitet.")


# ---------------------------------------------------------
# Argumentparser (akzeptiert optional argv)
# ---------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trainiert Word2Vec-Modelle auf Basis von korpus_gen-CSV-Dateien."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Ordner mit den Eingabe-CSV-Dateien (z. B. output/processed_corpus).",
    )
    parser.add_argument(
        "--pattern",
        default="korpus_gen*.csv",
        help="Dateinamen-Pattern für Eingabedateien (Standard: 'korpus_gen*.csv').",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Zielordner für gespeicherte Word2Vec-Modelle.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------
# Main (CLI-Wrapper)
# ---------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
    )


if __name__ == "__main__":
    main()