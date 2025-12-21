#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Erzeugt DTM- und TF-IDF-Matrizen sowie Cosinus-Matrizen für definierte
Zeitintervalle aus dem vorverarbeiteten Korpus 'korpus_stop.csv'.

ÄNDERUNG v3:
- Automatische Delimiter-Erkennung (Fallback: ";")
- Dynamische Intervall-Generierung aus Jahresdaten
- Arbeitet mit content_stop (nicht "content")
- Flexible Metadaten-Handhabung
- Konsistent mit Pipeline v3

Für jedes Zeitintervall:
    1) Auswahl der Dokumente nach Jahr (year_first hat Vorrang vor year)
    2) Erzeugung einer DTM (CountVectorizer, max_features=2000)
    3) Erzeugung einer TF-IDF-Matrix (TfidfVectorizer, max_features=2000)
    4) Berechnung der Cosinus-Ähnlichkeitsmatrix auf Basis der TF-IDF-Matrix
    5) Speicherung aller Matrizen als CSV

Beispielaufruf:

    python s05_dtm_tfidf_cos_intervals.py \\
        --input output/processed_corpus/korpus_stop.csv \\
        --dtm-output output/intervals/dtm_tfidf_stop \\
        --cos-output output/intervals/cosine_stop \\
        --sep auto
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import der gemeinsamen Utils
try:
    from .pipeline_utils import (
        detect_delimiter,
        identify_content_column,
        identify_metadata_columns,
        identify_id_column,
        identify_year_columns,
        get_year_series,
        detect_year_range,
        has_column, safe_filename
    )
except ImportError:
    from pipeline_utils import (
        detect_delimiter,
        identify_content_column,
        identify_metadata_columns,
        identify_id_column,
        identify_year_columns,
        get_year_series,
        detect_year_range,
        has_column, safe_filename
    )


# ---------------------------------------------------------
# Dynamische Intervall-Generierung
# ---------------------------------------------------------

def generate_intervals_from_data(
    df: pd.DataFrame,
    interval_size: int = 15
) -> List[Tuple[str, int, int]]:
    """
    Generiert Intervalle dynamisch aus den Jahresdaten.
    
    Returns:
        Liste von (label, start_year, end_year) Tupeln
    """
    year_range = detect_year_range(df)
    if year_range is None:
        return []
    
    min_year, max_year = year_range
    intervals = []
    
    start = min_year
    while start <= max_year:
        end = min(start + interval_size - 1, max_year)
        label = f"{start}-{end}"
        intervals.append((label, start, end))
        start = end + 1
    
    return intervals


MATRIX_TYPES = {
    "dtm-2000": CountVectorizer(max_features=2000),
    "tfidf-2000": TfidfVectorizer(max_features=2000),
}


# ---------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------

def load_corpus(path: Path, sep: str = "auto") -> Tuple[pd.DataFrame, str, str]:
    """
    Lädt das Korpus und bereitet year/year_first und content vor.
    
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

    # content auf String + fehlende Werte abfangen
    df[content_col] = df[content_col].fillna("").astype(str)

    # effective_year erstellen (year_first hat Vorrang)
    year_series = get_year_series(df)
    if year_series is not None:
        df["effective_year"] = year_series

    return df, content_col, sep


def subset_interval(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Filtert das DataFrame nach effective_year im gegebenen Intervall."""
    if "effective_year" not in df.columns:
        return pd.DataFrame()
    
    sub = df.copy()
    sub = sub[sub["effective_year"].notna()]
    sub = sub[(sub["effective_year"] >= start) & (sub["effective_year"] <= end)]

    if not sub.empty:
        sub = sub.copy()
        if "year" in sub.columns:
            sub["year"] = sub["effective_year"].astype(int)

    return sub


def create_matrix(
    df: pd.DataFrame,
    content_col: str,
    matrix_name: str,
    vectorizer,
) -> Tuple[pd.DataFrame, list[str]]:
    """Erzeugt eine DTM/TF-IDF-Matrix."""
    print(f"    ➡ Erzeuge Matrix: {matrix_name}")

    texts = df[content_col].fillna("").astype(str)
    if texts.str.strip().eq("").all():
        raise ValueError("Alle Texte in diesem Intervall sind leer.")

    V = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out().tolist()

    matrix_df = pd.DataFrame(V.toarray(), columns=terms)
    return matrix_df, terms


def save_dtm_with_metadata(
    df_interval: pd.DataFrame,
    content_col: str,
    matrix_df: pd.DataFrame,
    interval_name: str,
    matrix_name: str,
    text_field: str,
    out_dir: Path,
):
    """Speichert Metadaten + Matrix als CSV (OHNE Content-Spalte!)."""
    meta_cols = identify_metadata_columns(df_interval, content_col)
    available_meta_cols = [c for c in meta_cols if c in df_interval.columns and c != "effective_year"]
    
    if not available_meta_cols:
        out_df = matrix_df.reset_index(drop=True)
    else:
        meta_df = df_interval[available_meta_cols].reset_index(drop=True)
        out_df = pd.concat([meta_df, matrix_df.reset_index(drop=True)], axis=1)

    filename = f"{interval_name}_{matrix_name}_{text_field}.csv"
    out_path = out_dir / safe_filename(filename)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"    ✔ DTM/TF-IDF gespeichert: {out_path}")


def compute_and_save_cosine(
    matrix_df: pd.DataFrame,
    df_interval: pd.DataFrame,
    interval_name: str,
    matrix_name: str,
    text_field: str,
    out_dir: Path,
):
    """Berechnet und speichert die Cosinus-Ähnlichkeitsmatrix."""
    print(f"    ➡ Berechne Cosinus-Matrix für {interval_name}, {matrix_name} ...")

    features = matrix_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if features.isna().values.any():
        raise ValueError("âŒ Nach Bereinigung sind noch NaN in den TF-IDF-Features vorhanden.")

    M = features.to_numpy()
    if M.size == 0:
        raise ValueError("âŒ Leere Matrix für Cosinus-Berechnung.")

    cos = cosine_similarity(M)

    # Dokument-IDs bestimmen (flexibel)
    id_col = identify_id_column(df_interval)
    
    if id_col:
        doc_ids = df_interval[id_col].fillna("").astype(str).tolist()
    else:
        doc_ids = [f"doc_{i}" for i in range(len(df_interval))]

    cos_df = pd.DataFrame(cos, index=doc_ids, columns=doc_ids)

    filename = f"{interval_name}_cos_{matrix_name}_{text_field}.csv"
    out_path = out_dir / safe_filename(filename)
    out_dir.mkdir(parents=True, exist_ok=True)
    cos_df.to_csv(out_path, index=True, encoding="utf-8")

    print(f"    ✔ Cosinus-Matrix gespeichert: {out_path}")


# ---------------------------------------------------------
# Run-Funktion für Pipeline
# ---------------------------------------------------------

def run(
    input_path: Path,
    dtm_output: Path,
    cos_output: Path,
    sep: str = "auto",
    interval_size: int = 15,
) -> str:
    """
    Erzeugt DTM/TF-IDF- und Cosinus-Matrizen für Zeitintervalle.
    
    Returns:
        Verwendeter Delimiter
    """

    print(f"📄 Lade Korpus: {input_path}")
    df, content_col, used_sep = load_corpus(input_path, sep=sep)

    # Metadaten anzeigen
    metadata_cols = identify_metadata_columns(df, content_col)
    print(f"📋 Erkannte Metadaten: {len(metadata_cols)} Spalten")
    print(f"ℹ️  Content-Spalte ({content_col}) wird NICHT in den Matrizen gespeichert")

    # Prüfen ob year/year_first vorhanden
    year_first, year = identify_year_columns(df)
    if not year_first and not year:
        print("⚠️ Korpus enthält weder 'year' noch 'year_first'-Spalte.")
        print("   → Intervall-Verarbeitung wird übersprungen.")
        print("\n✅ Verarbeitung beendet (keine Intervalle möglich).")
        return used_sep

    # Intervalle dynamisch generieren
    intervals = generate_intervals_from_data(df, interval_size)
    print(f"📅 {len(intervals)} Intervalle generiert")

    text_field = "stop"

    for interval_name, start, end in intervals:
        print(f"\nâ³ Verarbeite Intervall {interval_name} ({start}–{end}) â€¦")

        df_interval = subset_interval(df, start, end)
        if df_interval.empty:
            print(f"    ⚠️  Keine Dokumente im Intervall {interval_name} gefunden.")
            continue

        print(f"    ✔ {len(df_interval)} Dokument(e) im Intervall {interval_name}.")

        # Für jedes Intervall: DTM-2000 und TF-IDF-2000
        tfidf_matrix_df = None
        for matrix_name, vectorizer in MATRIX_TYPES.items():
            try:
                matrix_df, terms = create_matrix(df_interval, content_col, matrix_name, vectorizer)
            except ValueError as e:
                print(f"    ⚠️  Übersprungen ({matrix_name}): {e}")
                continue

            save_dtm_with_metadata(
                df_interval=df_interval,
                content_col=content_col,
                matrix_df=matrix_df,
                interval_name=interval_name,
                matrix_name=matrix_name,
                text_field=text_field,
                out_dir=dtm_output,
            )

            if matrix_name == "tfidf-2000":
                tfidf_matrix_df = matrix_df

        # Cosinus nur für TF-IDF-2000
        if tfidf_matrix_df is not None:
            try:
                compute_and_save_cosine(
                    matrix_df=tfidf_matrix_df,
                    df_interval=df_interval,
                    interval_name=interval_name,
                    matrix_name="tfidf-2000",
                    text_field=text_field,
                    out_dir=cos_output,
                )
            except ValueError as e:
                print(f"    ⚠️  Cosinus-Berechnung übersprungen: {e}")

    print("\n✅ Alle Intervalle verarbeitet.")
    return used_sep


# ---------------------------------------------------------
# Argumente
# ---------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erzeugt DTM/TF-IDF- und Cosinus-Matrizen für Zeitintervalle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ÄNDERUNG v3:
  - Automatische Delimiter-Erkennung (--sep auto)
  - Dynamische Intervall-Generierung aus Jahresdaten
  - Konsistent mit Pipeline v3

Beispiel:
  python s05_dtm_tfidf_cos_intervals.py \\
      --input output/processed_corpus/korpus_stop.csv \\
      --dtm-output output/intervals/dtm_tfidf_stop \\
      --cos-output output/intervals/cosine_stop \\
      --sep auto
        """
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Pfad zur Eingabedatei (korpus_stop.csv mit content_stop).",
    )
    parser.add_argument(
        "--dtm-output",
        required=True,
        type=Path,
        help="Zielordner für DTM/TF-IDF-CSV-Dateien.",
    )
    parser.add_argument(
        "--cos-output",
        required=True,
        type=Path,
        help="Zielordner für Cosinus-CSV-Dateien.",
    )
    parser.add_argument(
        "--sep",
        default="auto",
        help="CSV-Delimiter ('auto' für automatische Erkennung).",
    )
    parser.add_argument(
        "--interval-size",
        type=int,
        default=15,
        help="Intervallgröße in Jahren (Standard: 15).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------
# Main (CLI-Wrapper)
# ---------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    run(
        input_path=args.input,
        dtm_output=args.dtm_output,
        cos_output=args.cos_output,
        sep=args.sep,
        interval_size=args.interval_size,
    )


if __name__ == "__main__":
    main()
