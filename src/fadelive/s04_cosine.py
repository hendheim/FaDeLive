#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Berechnung einer Cosinus-Ähnlichkeitsmatrix auf Grundlage der TF-IDF-2000-Matrix.

Input-Datei:
    output/dtm_tfidf_stop/tfidf-2000.csv

Output-Datei:
    output/cosine/cosine_tfidf2000.csv

Dieses Script:
    1) lädt die TF-IDF-2000-Matrix
    2) trennt Metadaten von Feature-Spalten
    3) berechnet die Cosinus-Ähnlichkeitsmatrix
    4) speichert die Cosinus-Matrix als CSV

Beispielaufruf:

    python src/fadelive/s04_cosine.py `
        --input output/dtm_tfidf_stop/tfidf-2000.csv `
        --output output/cosine/cosine_tfidf2000.csv

"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Funktionen
# ---------------------------------------------------------

def load_tfidf_matrix(path: Path):
    """Lädt TF-IDF-Matrix und trennt Metadaten von Feature-Spalten."""
    if not path.exists():
        raise FileNotFoundError(f"❌ Datei nicht gefunden: {path}")

    # Standard-CSV (Komma-getrennt)
    df = pd.read_csv(path, encoding="utf-8")

    # Feste Metadatenliste – alles andere wird als TF-IDF-Feature betrachtet
    METADATA = {
        "_id", "author_prename", "author_surname", "title", "source", "year",
        "editor_prename", "editor_surname", "volume", "title_addition",
        "year_first", "edition", "issue", "pages", "pages_exzerpt", "archive",
        "author_address", "address", "genre", "textclass", "note",
        "female_education", "author_address_geo", "address_geo"
    }

    # Welche Spalten sind Metadaten? Welche sind Features?
    meta_cols = df.columns.intersection(METADATA).tolist()
    feature_cols = df.columns.difference(METADATA).tolist()

    if not feature_cols:
        raise ValueError("❌ Keine TF-IDF-Feature-Spalten gefunden.")

    # *** HIER passiert die entscheidende Bereinigung ***
    # Alles, was Feature ist, numerisch erzwingen → nicht-numerisches wird NaN → dann zu 0
    features = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df[feature_cols] = features

    # Optional: Debug-Ausgabe, falls du noch Probleme hast
    # print("Feature-Spalten:", feature_cols[:10])
    # print(df[feature_cols].dtypes.head())

    return df, meta_cols, feature_cols

def compute_cosine(df_features: pd.DataFrame) -> pd.DataFrame:
    """Berechnet die Cosinus-Ähnlichkeitsmatrix."""
    # Sicherheit: falls vorher doch irgendwo NaN geblieben ist
    if df_features.isna().values.any():
        raise ValueError("❌ Nach Bereinigung sind noch NaN in den TF-IDF-Features vorhanden.")

    matrix = df_features.to_numpy(dtype=float)

    if matrix.size == 0:
        raise ValueError("❌ Die TF-IDF-Matrix enthält keine Daten.")

    cos = cosine_similarity(matrix)
    return pd.DataFrame(cos)

# ---------------------------------------------------------
# run-Funktion für Pipeline
# ---------------------------------------------------------

def run(
    input_path: Path,
    output_path: Path,
) -> None:
    """Berechnet eine Cosinusmatrix aus tfidf-2000.csv."""

    print(f"📄 Lade TF-IDF-Datei: {input_path}")
    df, meta_cols, feature_cols = load_tfidf_matrix(input_path)

    # Dokument-IDs bestimmen
    if "_id" in df.columns:
        doc_ids = df["_id"].fillna("").astype(str).tolist()
    else:
        doc_ids = [f"doc_{i}" for i in range(len(df))]

    print("➡ Berechne Cosinus-Ähnlichkeit …")
    df_cos = compute_cosine(df[feature_cols])

    # Spalten und Index beschriften
    df_cos.index = doc_ids
    df_cos.columns = doc_ids

    # Zielordner anlegen
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 Speichere Datei: {output_path}")
    df_cos.to_csv(output_path, encoding="utf-8", index=True)

    print("✅ Cosinusmatrix erfolgreich erstellt.")


# ---------------------------------------------------------
# Argumentparser
# ---------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Berechnet eine Cosinusmatrix aus tfidf-2000.csv."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Pfad zur TF-IDF-2000-Eingabedatei.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Pfad zur Cosinus-Ausgabedatei.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------
# MAIN – CLI-Wrapper
# ---------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    run(
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()