#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Erzeugt DTM- und TF-IDF-Matrizen aus dem vollständig
vorverarbeiteten Stopwort-Korpus 'korpus_stop.csv'.

ÄNDERUNG v3:
- Automatische Delimiter-Erkennung (Fallback: ";")
- Arbeitet mit content_stop (nicht "content")
- Flexible Metadaten-Handhabung: Alle Spalten außer content_stop werden als Metadaten behandelt
- OUTPUT: Nur Metadaten + Features (KEINE Content-Spalte in den Matrizen!)
- Konsistent mit Pipeline v3

Eingabe:
    output/processed_corpus/korpus_stop.csv (mit content_stop)

Ausgaben:
    output/dtm_tfidf_stop/
        dtm-500.csv (Metadaten + 500 Features)
        dtm-1000.csv (Metadaten + 1000 Features)
        dtm-2000.csv (Metadaten + 2000 Features)
        tfidf-500.csv (Metadaten + 500 Features)
        tfidf-1000.csv (Metadaten + 1000 Features)
        tfidf-2000.csv (Metadaten + 2000 Features)
        dtm_minfreq6.csv (Metadaten + Features mit min. 6 Vorkommen)

Beispielaufruf:

    python s03_dtm_tfidf.py \\
        --input output/processed_corpus/korpus_stop.csv \\
        --output output/dtm_tfidf_stop \\
        --sep auto
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Import der gemeinsamen Utils
try:
    from .pipeline_utils import (
        detect_delimiter,
        identify_content_column,
        identify_metadata_columns,
        has_column, safe_filename
    )
except ImportError:
    from pipeline_utils import (
        detect_delimiter,
        identify_content_column,
        identify_metadata_columns,
        has_column, safe_filename
    )


# ---------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------

def load_corpus(path: Path, sep: str = "auto") -> tuple[pd.DataFrame, str, str]:
    """
    Lädt das Korpus und entfernt alle Dokumente ohne echten Inhalt.
    
    Returns:
        (DataFrame, content_column_name, delimiter)
    """

    if not path.exists():
        raise FileNotFoundError(f"âŒ Eingabedatei nicht gefunden: {path}")

    # Delimiter erkennen
    if sep == "auto":
        sep = detect_delimiter(path)

    df = pd.read_csv(path, sep=sep, encoding="utf-8")

    # Content-Spalte identifizieren
    content_col = identify_content_column(df)
    if content_col is None:
        raise ValueError("Keine Content-Spalte gefunden")
    print(f"📋 Erkannte Content-Spalte: {content_col}")

    # content normalisieren
    df[content_col] = df[content_col].fillna("").astype(str)

    # Entferne Dokumente ohne Inhalt
    def has_real_text(s: str) -> bool:
        s_clean = "".join([c for c in s if c.isalpha()])
        return bool(s_clean.strip())

    before = len(df)
    df = df[df[content_col].apply(has_real_text)].copy()
    after = len(df)

    dropped = before - after

    if after == 0:
        raise ValueError("âŒ Kein einziges Dokument enthält verwertbaren Inhalt.")

    if dropped > 0:
        print(f"⚠️  {dropped} Dokument(e) wegen fehlendem Inhalt übersprungen.")

    return df, content_col, sep


def save_matrix(df_meta: pd.DataFrame, matrix, terms, out_file: Path):
    """
    Kombiniert Metadaten + Matrix und speichert sie.
    
    WICHTIG: Keine Content-Spalte in der Ausgabe!
    """
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

def create_matrix(
    df: pd.DataFrame,
    content_col: str,
    metadata_cols: list[str],
    name: str,
    vectorizer,
    output_dir: Path
):
    """Berechnet eine DTM/TF-IDF und speichert sie (ohne Content-Spalte!)."""
    print(f"➡ Erzeuge Matrix: {name}")

    if df.empty:
        print(f"⚠️  Übersprungen: Kein Dokument mit Inhalt (Matrix {name}).")
        return

    V = vectorizer.fit_transform(df[content_col])
    if V.shape[1] == 0:
        print(f"⚠️  Keine Terme für {name}. Matrix wird übersprungen.")
        return

    terms = vectorizer.get_feature_names_out()
    matrix = V.toarray()

    # Nur vorhandene Metadaten verwenden (OHNE Content!)
    available_meta_cols = [c for c in metadata_cols if c in df.columns]
    df_meta = df[available_meta_cols].copy()

    out_file = output_dir / f"{safe_filename(name)}.csv"
    save_matrix(df_meta, matrix, terms, out_file)


def create_frequency_based_matrix(
    df: pd.DataFrame,
    content_col: str,
    metadata_cols: list[str],
    min_freq: int,
    output_dir: Path
):
    """Erzeugt eine DTM aller Wörter, die mindestens min_freq Vorkommen haben."""
    print(f"➡ Erzeuge DTM (min. {min_freq} Vorkommen)")

    vec = CountVectorizer()
    V = vec.fit_transform(df[content_col])

    terms = vec.get_feature_names_out()
    freqs = V.toarray().sum(axis=0)

    freq_df = pd.DataFrame({"term": terms, "freq": freqs})
    selected_terms = freq_df[freq_df["freq"] >= min_freq]["term"].tolist()

    if not selected_terms:
        print(f"⚠️  Keine Terme erfüllen die Bedingung â‰¥ {min_freq}. Übersprungen.")
        return

    full_matrix = pd.DataFrame(V.toarray(), columns=terms)
    filtered_matrix = full_matrix[selected_terms]

    # Nur vorhandene Metadaten verwenden (OHNE Content!)
    available_meta_cols = [c for c in metadata_cols if c in df.columns]
    df_meta = df[available_meta_cols].copy()
    
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
    sep: str = "auto",
) -> str:
    """
    Erstellt DTM- und TF-IDF-Matrizen aus einem Stopwort-Korpus.
    
    Returns:
        Verwendeter Delimiter
    """

    print(f"📄 Lade Korpus: {input_path}")
    df, content_col, used_sep = load_corpus(input_path, sep=sep)

    # Metadaten automatisch erkennen (ohne Content!)
    metadata_cols = identify_metadata_columns(df, content_col)
    print(f"📋 Erkannte Metadaten-Spalten: {len(metadata_cols)}")
    print(f"ℹ️  Content-Spalte ({content_col}) wird NICHT in den Matrizen gespeichert")

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
        create_matrix(df, content_col, metadata_cols, name, vec, output_dir)

    create_frequency_based_matrix(df, content_col, metadata_cols, min_freq=6, output_dir=output_dir)

    print("\n✅ Alle Matrizen wurden erfolgreich erzeugt.")
    return used_sep


# ---------------------------------------------------------
# Argumentparser
# ---------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Erstellt DTM- und TF-IDF-Matrizen aus einem Stopwort-Korpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ÄNDERUNG v3:
  - Automatische Delimiter-Erkennung (--sep auto)
  - Flexible Content-/Metadaten-Erkennung
  - Konsistent mit Pipeline v3

Beispiel:
  python s03_dtm_tfidf.py \\
      --input output/processed_corpus/korpus_stop.csv \\
      --output output/dtm_tfidf_stop \\
      --sep auto
        """
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
        default="auto",
        help="CSV-Delimiter ('auto' für automatische Erkennung)",
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
