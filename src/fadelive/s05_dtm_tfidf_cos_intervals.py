#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Erzeugt DTM- und TF-IDF-Matrizen sowie Cosinus-Matrizen für definierte
Zeitintervalle aus dem vorverarbeiteten Korpus 'korpus_stop.csv'.

ÄNDERUNG v2:
- Arbeitet mit content_stop (nicht "content")
- Flexible Metadaten-Handhabung: Alle Spalten außer content_stop werden als Metadaten behandelt
- OUTPUT: DTM/TF-IDF enthalten nur Metadaten + Features (KEINE Content-Spalte!)
- year/year_first werden speziell für Intervall-Filterung verwendet (year_first hat Vorrang)
- Automatische Trennung von Metadaten und Features bei Cosinus-Berechnung

Basis:
    - Eingabekorpus enthält Stopwort-bereinigte Texte in der Spalte 'content_stop'
    - Metadaten (flexibel, besonders year/year_first für Intervalle)

Für jedes Zeitintervall:
    1) Auswahl der Dokumente nach Jahr (year_first hat Vorrang vor year)
    2) Erzeugung einer DTM (CountVectorizer, max_features=2000)
    3) Erzeugung einer TF-IDF-Matrix (TfidfVectorizer, max_features=2000)
    4) Berechnung der Cosinus-Ähnlichkeitsmatrix auf Basis der TF-IDF-Matrix
    5) Speicherung aller Matrizen als CSV

Eingabe (Standardidee):
    output/processed_corpus/korpus_stop.csv (mit content_stop)

Ausgaben (Beispiel):
    output/intervals/dtm_tfidf_stop/
        1782-1852_dtm-2000_stop.csv (Metadaten + Features, keine Content-Spalte!)
        1782-1852_tfidf-2000_stop.csv (Metadaten + Features, keine Content-Spalte!)
        ...
    output/intervals/cosine_stop/
        1782-1852_cos_tfidf-2000_stop.csv (nur Cosinus-Matrix mit Doc-IDs)
        ...

Beispielaufruf:

    python s05_dtm_tfidf_cos_intervals_v2.py \
        --input output/processed_corpus/korpus_stop.csv \
        --dtm-output output/intervals/dtm_tfidf_stop \
        --cos-output output/intervals/cosine_stop \
        --sep ";"
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Zeitintervalle
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
    "1865-1891": (1865, 1891),
}

MATRIX_TYPES = {
    "dtm-2000": CountVectorizer(max_features=2000),
    "tfidf-2000": TfidfVectorizer(max_features=2000),
}


# ---------------------------------------------------------
# Metadaten-Erkennung
# ---------------------------------------------------------

def identify_content_column(df: pd.DataFrame) -> str:
    """
    Identifiziert die Content-Spalte (content_stop, content_lem, content_min oder content_gen).
    
    Returns:
        Name der Content-Spalte
    """
    for col in ["content_stop", "content_lem", "content_min", "content_gen"]:
        if col in df.columns:
            return col
    raise ValueError("Keine Content-Spalte gefunden (content_stop/content_lem/content_min/content_gen)")


def identify_metadata_columns(df: pd.DataFrame, content_col: str) -> list[str]:
    """Identifiziert alle Metadaten-Spalten (= alles außer der Content-Spalte)."""
    return [col for col in df.columns if col != content_col]


def has_column(df: pd.DataFrame, col: str) -> bool:
    """Prüft, ob eine Spalte existiert und nicht-leere Werte enthält."""
    return col in df.columns and df[col].notna().any()


def identify_metadata_and_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Trennt automatisch Metadaten von TF-IDF-Features.
    
    Strategie:
    - Metadaten = nicht-numerische Spalten ODER bekannte Metadaten-Namen
    - Features = numerische Spalten, die nicht zu bekannten Metadaten gehören
    
    Returns:
        (metadata_columns, feature_columns)
    """
    # Bekannte Metadaten-Namen
    KNOWN_METADATA = {
        "_id", "id", "author_prename", "author_surname", "title", "source", "year",
        "editor_prename", "editor_surname", "volume", "title_addition",
        "year_first", "edition", "issue", "pages", "pages_exzerpt", "archive",
        "author_address", "address", "genre", "textclass", "note",
        "female_education", "author_address_geo", "address_geo"
    }
    
    metadata_cols = []
    feature_cols = []
    
    for col in df.columns:
        if col in KNOWN_METADATA:
            metadata_cols.append(col)
        elif not pd.api.types.is_numeric_dtype(df[col]):
            metadata_cols.append(col)
        else:
            feature_cols.append(col)
    
    return metadata_cols, feature_cols


# ---------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------

def safe_filename(s: str) -> str:
    """Sichere Dateinamen generieren."""
    return str(s).replace(" ", "_").replace("/", "_").replace("\\", "_")


def load_corpus(path: Path, sep: str = ",") -> tuple[pd.DataFrame, str]:
    """
    Lädt das Korpus und bereitet year/year_first und content vor.
    
    Returns:
        (DataFrame, content_column_name)
    """
    if not path.exists():
        raise FileNotFoundError(f"❌ Eingabedatei nicht gefunden: {path}")

    df = pd.read_csv(path, sep=sep, encoding="utf-8")

    # Content-Spalte identifizieren
    content_col = identify_content_column(df)
    print(f"📋 Erkannte Content-Spalte: {content_col}")

    # content auf String + fehlende Werte abfangen
    df[content_col] = df[content_col].fillna("").astype(str)

    # Jahrspalten in Zahlen konvertieren
    if "year" in df.columns:
        year = pd.to_numeric(df["year"], errors="coerce")
    else:
        year = pd.Series(dtype=float)
    
    if "year_first" in df.columns:
        year_first = pd.to_numeric(df["year_first"], errors="coerce")
    else:
        year_first = pd.Series(dtype=float)

    # effective_year: year_first hat Vorrang, sonst year
    df["effective_year"] = year_first.combine_first(year)

    return df, content_col


def subset_interval(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Filtert das DataFrame nach effective_year im gegebenen Intervall."""
    sub = df.copy()
    sub = sub[sub["effective_year"].notna()]
    sub = sub[(sub["effective_year"] >= start) & (sub["effective_year"] <= end)]

    # year-Spalte auf effective_year setzen (für die Ausgabe)
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
) -> tuple[pd.DataFrame, list[str]]:
    """
    Erzeugt eine DTM/TF-IDF-Matrix aus df[content_col] und gibt
    (Matrix-DataFrame, Feature-Liste) zurück.
    """
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
    """
    Speichert Metadaten + Matrix als CSV (OHNE Content-Spalte!).
    """
    # Metadaten-Spalten automatisch erkennen (ohne Content!)
    meta_cols = identify_metadata_columns(df_interval, content_col)
    
    # Nur vorhandene Metadaten verwenden (ohne effective_year)
    available_meta_cols = [c for c in meta_cols if c in df_interval.columns and c != "effective_year"]
    
    if not available_meta_cols:
        # Fallback: nur Matrix speichern
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
    print(f"    ➡ Berechne Cosinus-Matrix für {interval_name}, {matrix_name} …")

    # Sicherstellen, dass alles numerisch ist
    features = matrix_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if features.isna().values.any():
        raise ValueError("❌ Nach Bereinigung sind noch NaN in den TF-IDF-Features vorhanden.")

    M = features.to_numpy()
    if M.size == 0:
        raise ValueError("❌ Leere Matrix für Cosinus-Berechnung.")

    cos = cosine_similarity(M)

    # Dokument-IDs bestimmen (flexibel)
    id_col = None
    for possible_id in ["_id", "id"]:
        if possible_id in df_interval.columns:
            id_col = possible_id
            break
    
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
    sep: str = ",",
) -> None:
    """Erzeugt DTM/TF-IDF- und Cosinus-Matrizen für Zeitintervalle aus korpus_stop."""

    print(f"📄 Lade Korpus: {input_path}")
    df, content_col = load_corpus(input_path, sep=sep)

    # Metadaten anzeigen
    metadata_cols = identify_metadata_columns(df, content_col)
    print(f"📋 Erkannte Metadaten: {', '.join(metadata_cols)}")
    print(f"ℹ️  Content-Spalte ({content_col}) wird NICHT in den Matrizen gespeichert")

    # Prüfen ob year/year_first vorhanden
    if not (has_column(df, "year") or has_column(df, "year_first")):
        raise ValueError("❌ Korpus enthält weder 'year' noch 'year_first'-Spalte. Intervalle können nicht angewendet werden.")

    text_field = "stop"  # historisch konsistent benannt

    for interval_name, (start, end) in INTERVALS.items():
        print(f"\n⏳ Verarbeite Intervall {interval_name} ({start}–{end}) …")

        df_interval = subset_interval(df, start, end)
        if df_interval.empty:
            print(f"    ⚠️  Keine Dokumente im Intervall {interval_name} gefunden.")
            continue

        print(f"    ✔ {len(df_interval)} Dokument(e) im Intervall {interval_name}.")

        # Für jedes Intervall: DTM-2000 und TF-IDF-2000
        tfidf_matrix_df = None  # merken für Cosinus
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


# ---------------------------------------------------------
# Argumente
# ---------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erzeugt DTM/TF-IDF- und Cosinus-Matrizen für Zeitintervalle aus korpus_stop."
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
        default=",",
        help="CSV-Delimiter der Eingabedatei (Standard: ',').",
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
    )


if __name__ == "__main__":
    main()
