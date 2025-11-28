#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Erzeugt DTM- und TF-IDF-Matrizen aus dem vollständig
vorverarbeiteten Stopwort-Korpus 'korpus_stop.csv'.

Dokumente mit leerem oder nur trivialem Inhalt werden *nicht* berücksichtigt.

Eingabe:
    output/processed_corpus/korpus_stop.csv

Ausgaben:
    output/dtm_tfidf_stop/
        dtm-500.csv
        dtm-1000.csv
        dtm-2000.csv
        tfidf-500.csv
        tfidf-1000.csv
        tfidf-2000.csv
        dtm_minfreq6.csv

Beispielaufruf:

    python src/fadelive/s03_dtm_tfidf.py `
        --input output/processed_corpus/korpus_stop.csv `
        --output output/dtm_tfidf_stop `
        --sep ";"

"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ---------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------

def safe_filename(name: str) -> str:
    """Erzeugt sichere Dateinamen."""
    return str(name).replace(" ", "_").replace("/", "_").replace("\\", "_")


def load_corpus(path: Path, sep: str = ",") -> pd.DataFrame:
    """Lädt das Korpus und entfernt alle Dokumente ohne echten Inhalt."""

    if not path.exists():
        raise FileNotFoundError(f"❌ Eingabedatei nicht gefunden: {path}")

    df = pd.read_csv(path, sep=sep, encoding="utf-8")

    if "content" not in df.columns:
        raise ValueError("❌ Spalte 'content' fehlt im Korpus.")

    # content normalisieren
    df["content"] = df["content"].fillna("").astype(str)

    # Entferne Dokumente ohne Inhalt:
    # - leer
    # - nur Whitespace
    # - nur Sonderzeichen/Zahlen
    def has_real_text(s: str) -> bool:
        s_clean = "".join([c for c in s if c.isalpha()])
        return bool(s_clean.strip())

    before = len(df)
    df = df[df["content"].apply(has_real_text)].copy()
    after = len(df)

    dropped = before - after

    if after == 0:
        raise ValueError("❌ Kein einziges Dokument enthält verwertbaren Inhalt.")

    if dropped > 0:
        print(f"⚠ {dropped} Dokument(e) wegen fehlendem Inhalt übersprungen.")

    return df


def save_matrix(df_meta: pd.DataFrame, matrix, terms, out_file: Path):
    """Kombiniert Metadaten + Matrix und speichert sie."""
    df_matrix = pd.DataFrame(matrix, columns=terms)

    df_out = pd.concat(
        [df_meta.reset_index(drop=True),
         df_matrix.reset_index(drop=True)],
        axis=1
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_file, index=False, encoding="utf-8")
    print(f"✔ Gespeichert: {out_file}")


# ---------------------------------------------------------
# Matrizen erzeugen
# ---------------------------------------------------------

def create_matrix(df: pd.DataFrame, name: str, vectorizer, output_dir: Path):
    """Berechnet eine DTM/TF-IDF und speichert sie."""
    print(f"➡ Erzeuge Matrix: {name}")

    if df.empty:
        print(f"⚠ Übersprungen: Kein Dokument mit Inhalt (Matrix {name}).")
        return

    V = vectorizer.fit_transform(df["content"])
    if V.shape[1] == 0:
        print(f"⚠ Keine Terme für {name}. Matrix wird übersprungen.")
        return

    terms = vectorizer.get_feature_names_out()
    matrix = V.toarray()

    meta_cols = [c for c in df.columns if c != "content"]
    df_meta = df[meta_cols].copy()

    out_file = output_dir / f"{safe_filename(name)}.csv"
    save_matrix(df_meta, matrix, terms, out_file)


def create_frequency_based_matrix(df: pd.DataFrame, min_freq: int, output_dir: Path):
    """Erzeugt eine DTM aller Wörter, die mindestens min_freq Vorkommen haben."""
    print(f"➡ Erzeuge DTM (min. {min_freq} Vorkommen)")

    vec = CountVectorizer()
    V = vec.fit_transform(df["content"])

    terms = vec.get_feature_names_out()
    freqs = V.toarray().sum(axis=0)

    freq_df = pd.DataFrame({"term": terms, "freq": freqs})
    selected_terms = freq_df[freq_df["freq"] >= min_freq]["term"].tolist()

    if not selected_terms:
        print(f"⚠ Keine Terme erfüllen die Bedingung ≥ {min_freq}. Übersprungen.")
        return

    full_matrix = pd.DataFrame(V.toarray(), columns=terms)
    filtered_matrix = full_matrix[selected_terms]

    df_meta = df.drop(columns=["content"])
    df_out = pd.concat([df_meta.reset_index(drop=True),
                        filtered_matrix.reset_index(drop=True)], axis=1)

    out_file = output_dir / f"dtm_minfreq{min_freq}.csv"
    df_out.to_csv(out_file, index=False, encoding="utf-8")

    print(f"✔ Gespeichert: {out_file}")

# ---------------------------------------------------------
# run-Funktion für Pipeline
# ---------------------------------------------------------

def run(
    input_path: Path,
    output_dir: Path,
    sep: str = ",",
) -> None:
    """Erstellt DTM- und TF-IDF-Matrizen aus einem Stopwort-Korpus."""

    print(f"📄 Lade Korpus: {input_path}")
    df = load_corpus(input_path, sep=sep)

    vectorizers = {
        "dtm-500": CountVectorizer(max_features=500),
        "dtm-1000": CountVectorizer(max_features=1000),
        "dtm-2000": CountVectorizer(max_features=2000),
        "tfidf-500": TfidfVectorizer(max_features=500),
        "tfidf-1000": TfidfVectorizer(max_features=1000),
        "tfidf-2000": TfidfVectorizer(max_features=2000),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for name, vec in vectorizers.items():
        create_matrix(df, name, vec, output_dir)

    create_frequency_based_matrix(df, min_freq=6, output_dir=output_dir)

    print("\n✅ Alle Matrizen wurden erfolgreich erzeugt.")


# ---------------------------------------------------------
# Argumentparser
# ---------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Erstellt DTM- und TF-IDF-Matrizen aus einem Stopwort-Korpus."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Pfad zur korpus_stop.csv",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Zielordner für die Ausgabedateien",
    )
    parser.add_argument(
        "--sep",
        default=",",
        help="CSV-Delimiter (Standard ',')",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------
# MAIN – CLI-Wrapper
# ---------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    run(
        input_path=args.input,
        output_dir=args.output,
        sep=args.sep,
    )


if __name__ == "__main__":
    main()