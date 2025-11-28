#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Erzeugt DTM- und TF-IDF-Matrizen sowie Cosinus-Matrizen für definierte
Zeitintervalle aus dem vorverarbeiteten Korpus 'korpus_stop.csv'.

Basis:
    - Eingabekorpus enthält Stopwort-bereinigte Texte in der Spalte 'content'
    - Metadaten wie _id, year, year_first, author_surname, title, etc.

Für jedes Zeitintervall:
    1) Auswahl der Dokumente nach Jahr (year_first hat Vorrang vor year)
    2) Erzeugung einer DTM (CountVectorizer, max_features=2000)
    3) Erzeugung einer TF-IDF-Matrix (TfidfVectorizer, max_features=2000)
    4) Berechnung der Cosinus-Ähnlichkeitsmatrix auf Basis der TF-IDF-Matrix
    5) Speicherung aller Matrizen als CSV

Eingabe (Standardidee):
    output/processed_corpus/korpus_stop.csv

Ausgaben (Beispiel):
    output/intervals/dtm/
        1782-1852_dtm-2000_stop.csv
        1782-1852_tfidf-2000_stop.csv
        ...
    output/intervals/cosine/
        1782-1852_cos_tfidf-2000_stop.csv
        ...

Beispielaufruf:

    python src/fadelive/s05_dtm_tfidf_cos_intervals.py `
        --input output/processed_corpus/korpus_stop.csv `
        --dtm-output output/intervals/dtm_tfidf_stop `
        --cos-output output/intervals/cosine_stop `
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

# Metadaten, die auf jeden Fall mit in die DTM/TF-IDF-Datei sollen
DEFAULT_METADATA_FIELDS = [
    "_id",
    "author_surname",
    "title",
    "year",
    "source",
    "genre",
    "author_address",
    "address",
    "textclass",
]


# ---------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------

def safe_filename(s: str) -> str:
    """Sichere Dateinamen generieren."""
    return str(s).replace(" ", "_").replace("/", "_").replace("\\", "_")


def load_corpus(path: Path, sep: str = ",") -> pd.DataFrame:
    """Lädt das Korpus und bereitet year/year_first und content vor."""
    if not path.exists():
        raise FileNotFoundError(f"❌ Eingabedatei nicht gefunden: {path}")

    df = pd.read_csv(path, sep=sep, encoding="utf-8")

    if "content" not in df.columns:
        raise ValueError("❌ Spalte 'content' fehlt im Korpus.")

    # content auf String + fehlende Werte abfangen
    df["content"] = df["content"].fillna("").astype(str)

    # Jahrspalten in Zahlen konvertieren
    year = pd.to_numeric(df.get("year"), errors="coerce")
    year_first = pd.to_numeric(df.get("year_first"), errors="coerce")

    # effective_year: year_first hat Vorrang, sonst year
    df["effective_year"] = year_first.combine_first(year)

    return df


def subset_interval(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Filtert das DataFrame nach effective_year im gegebenen Intervall."""
    sub = df.copy()
    sub = sub[sub["effective_year"].notna()]
    sub = sub[(sub["effective_year"] >= start) & (sub["effective_year"] <= end)]

    # year-Spalte auf effective_year setzen (für die Ausgabe)
    if not sub.empty:
        sub = sub.copy()
        sub["year"] = sub["effective_year"].astype(int)

    return sub


def create_matrix(df: pd.DataFrame, matrix_name: str, vectorizer, text_field: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Erzeugt eine DTM/TF-IDF-Matrix aus df[text_field] und gibt
    (Matrix-DataFrame, Feature-Liste) zurück.
    """
    print(f"    ➡ Erzeuge Matrix: {matrix_name}")

    texts = df[text_field].fillna("").astype(str)
    if texts.str.strip().eq("").all():
        raise ValueError("Alle Texte in diesem Intervall sind leer.")

    V = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out().tolist()

    matrix_df = pd.DataFrame(V.toarray(), columns=terms)
    return matrix_df, terms


def save_dtm_with_metadata(
    df_interval: pd.DataFrame,
    matrix_df: pd.DataFrame,
    interval_name: str,
    matrix_name: str,
    text_field: str,
    out_dir: Path,
):
    """Speichert Metadaten + Matrix als CSV."""
    # Metadaten-Spalten bestimmen
    meta_cols = [c for c in DEFAULT_METADATA_FIELDS if c in df_interval.columns]
    if not meta_cols:
        # Fallback: alle Spalten außer content und effective_year
        meta_cols = [c for c in df_interval.columns if c not in ["content", "effective_year"]]

    meta_df = df_interval[meta_cols].reset_index(drop=True)
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

    # Dokument-IDs bestimmen (falls _id existiert, sonst laufende Nummern)
    if "_id" in df_interval.columns:
        doc_ids = df_interval["_id"].fillna("").astype(str).tolist()
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
    df = load_corpus(input_path, sep=sep)

    text_field = "content"

    for interval_name, (start, end) in INTERVALS.items():
        print(f"\n⏳ Verarbeite Intervall {interval_name} ({start}–{end}) …")

        df_interval = subset_interval(df, start, end)
        if df_interval.empty:
            print(f"    ⚠ Keine Dokumente im Intervall {interval_name} gefunden.")
            continue

        print(f"    ✔ {len(df_interval)} Dokument(e) im Intervall {interval_name}.")

        # Für jedes Intervall: DTM-2000 und TF-IDF-2000
        tfidf_matrix_df = None  # merken für Cosinus
        for matrix_name, vectorizer in MATRIX_TYPES.items():
            try:
                matrix_df, terms = create_matrix(df_interval, matrix_name, vectorizer, text_field)
            except ValueError as e:
                print(f"    ⚠ Übersprungen ({matrix_name}): {e}")
                continue

            save_dtm_with_metadata(
                df_interval=df_interval,
                matrix_df=matrix_df,
                interval_name=interval_name,
                matrix_name=matrix_name,
                text_field="stop",  # historisch konsistent benannt
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
                    text_field="stop",
                    out_dir=cos_output,
                )
            except ValueError as e:
                print(f"    ⚠ Cosinus-Berechnung übersprungen: {e}")

    print("\n✅ Alle Intervalle verarbeitet.")


# ---------------------------------------------------------
# Argumente (jetzt mit argv-Parameter)
# ---------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erzeugt DTM/TF-IDF- und Cosinus-Matrizen für Zeitintervalle aus korpus_stop."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Pfad zur Eingabedatei (korpus_stop.csv).",
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