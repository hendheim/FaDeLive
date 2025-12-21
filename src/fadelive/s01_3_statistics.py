#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistik-Pipeline für den Korpus.

ÄNDERUNG v3:
- Automatische Delimiter-Erkennung (Fallback: ";")
- Automatische Content-Spalten-Erkennung (content_min, content_lem, content_stop)
- Flexible Jahr-Erkennung (year_first, year, Jahr_final, jahr)
- Flexible ID-Erkennung (doc_id, _id, id, filename)
- Konsistent mit Pipeline v3

Eingabe:
    output/processed_corpus/korpus_min.csv (mit content_min)
    output/processed_corpus/korpus_lem.csv (mit content_lem)
    output/processed_corpus/korpus_stop.csv (mit content_stop)

Ausgabe:
    CSV-Dateien in output/statistics/, je nach vorhandenen Metadaten

Beispielaufruf:

    python s01_3_statistics.py \\
        --preprocessed-dir output/processed_corpus \\
        --output-dir output/statistics \\
        --delimiter auto
"""

import argparse
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, List, Tuple

import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

# Import der gemeinsamen Utils
try:
    from .pipeline_utils import (
        detect_delimiter, read_csv_auto,
        identify_content_column, identify_content_column_strict,
        identify_metadata_columns, identify_id_column,
        identify_year_columns, get_year_series, coalesce_years,
        has_column, safe_filename
    )
except ImportError:
    from pipeline_utils import (
        detect_delimiter, read_csv_auto,
        identify_content_column, identify_content_column_strict,
        identify_metadata_columns, identify_id_column,
        identify_year_columns, get_year_series, coalesce_years,
        has_column, safe_filename
    )


# =============================================================================
# NLTK vorbereiten
# =============================================================================

def ensure_nltk():
    """Stellt sicher, dass die notwendigen NLTK-Ressourcen vorhanden sind."""
    try:
        word_tokenize("Test")
    except LookupError:
        nltk.download("punkt")


def count_tokens(text: str) -> int:
    """Zählt Tokens in einem Text mit NLTK."""
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(word_tokenize(text))


# =============================================================================
# Laden der Korpora
# =============================================================================

def load_corpus_files(preprocessed_dir: Path, delimiter: str = "auto") -> Tuple[dict, str]:
    """
    Lädt korpus_min/lem/stop.csv, falls vorhanden.
    
    Args:
        preprocessed_dir: Ordner mit den Korpus-Dateien
        delimiter: CSV-Delimiter ("auto" für automatische Erkennung)
    
    Returns:
        (dict: {"min": (df, content_col), ...}, verwendeter_delimiter)
    """
    corpora = {}
    detected_delimiter = None
    
    for variant in ("min", "lem", "stop"):
        path = preprocessed_dir / f"korpus_{variant}.csv"
        if path.exists():
            print(f"   📄 Lade {path.name}")
            
            # Delimiter nur einmal erkennen
            if delimiter == "auto" and detected_delimiter is None:
                detected_delimiter = detect_delimiter(path)
            
            use_delimiter = detected_delimiter if delimiter == "auto" else delimiter
            df = pd.read_csv(path, sep=use_delimiter, encoding="utf-8")
            
            # Content-Spalte automatisch erkennen
            content_col = identify_content_column(df)
            if content_col is None:
                print(f"      ⚠️ Keine Content-Spalte gefunden in {path.name}, übersprungen.")
                continue
            
            print(f"      ✔ Content-Spalte: {content_col}")
            
            # Jahr-Spalten zusammenführen (falls vorhanden)
            year_first, year = identify_year_columns(df)
            if year_first or year:
                df = coalesce_years(df)
                print(f"      ✔ Jahr-Spalten: year_first={year_first or '—'}, year={year or '—'} → year_final")
            
            corpora[variant] = (df, content_col)
        else:
            print(f"   ⚠️ {path.name} nicht gefunden, übersprungen.")
    
    if not corpora:
        raise FileNotFoundError(f"Keine korpus_*.csv in {preprocessed_dir} gefunden.")
    
    final_delimiter = detected_delimiter if detected_delimiter else delimiter
    return corpora, final_delimiter


# =============================================================================
# Statistik-Funktionen
# =============================================================================

def compute_author_statistics(df_meta: pd.DataFrame, out_dir: Path):
    """Erstellt Author-Statistiken (falls author_surname vorhanden)."""
    if not has_column(df_meta, "author_surname"):
        print("   ⚠️ Keine 'author_surname' → author_statistics.csv übersprungen.")
        return
        
    df = df_meta.copy()
    df = df[df["author_surname"].astype(str).str.strip() != ""]
    stats = (
        df.groupby("author_surname", dropna=True)
        .size()
        .reset_index(name="anzahl_texte")
        .sort_values("anzahl_texte", ascending=False)
    )
    out_path = out_dir / "author_statistics.csv"
    stats.to_csv(out_path, index=False, encoding="utf-8")
    print(f"   ✅ author_statistics.csv")


def compute_token_statistics(corpora: dict, out_dir: Path):
    """Berechnet Token-Statistiken über alle Varianten."""
    tokens_rows = []
    tokens_per_tc_rows = []

    for variant, (df, content_col) in corpora.items():
        variant_name = variant
        df = df.copy()

        df["__tokens"] = df[content_col].astype(str).apply(count_tokens)

        total_tokens = int(df["__tokens"].sum())
        tokens_rows.append({"field": variant_name, "count": total_tokens})

        if has_column(df, "textclass"):
            grouped = (
                df.groupby("textclass", dropna=True)["__tokens"]
                .sum()
                .reset_index()
                .rename(columns={"__tokens": "count"})
            )
            for _, row in grouped.iterrows():
                tokens_per_tc_rows.append(
                    {
                        "textclass": row["textclass"],
                        "field": variant_name,
                        "count": int(row["count"]),
                    }
                )

    df_tokens = pd.DataFrame(tokens_rows)
    df_tokens.to_csv(out_dir / "tokens.csv", index=False, encoding="utf-8")
    print(f"   ✅ tokens.csv")

    if tokens_per_tc_rows:
        df_tokens_tc = pd.DataFrame(tokens_per_tc_rows)
        df_tokens_tc.to_csv(
            out_dir / "tokens_per_textclass.csv", index=False, encoding="utf-8"
        )
        print(f"   ✅ tokens_per_textclass.csv")
    else:
        print("   ⚠️️ Keine 'textclass' → tokens_per_textclass.csv übersprungen.")


def compute_textclass_and_documents(df_meta: pd.DataFrame, out_dir: Path):
    """Erstellt Document-Count und Textclass-Statistiken."""
    total_docs = len(df_meta)
    df_total = pd.DataFrame([{"total_documents": total_docs}])
    df_total.to_csv(out_dir / "documents_count.csv", index=False, encoding="utf-8")
    print(f"   ✅ documents_count.csv (n={total_docs})")

    if has_column(df_meta, "textclass"):
        df_tc = (
            df_meta.groupby("textclass", dropna=True)
            .size()
            .reset_index(name="count")
        )
        df_tc.to_csv(out_dir / "textclass_count.csv", index=False, encoding="utf-8")
        print(f"   ✅ textclass_count.csv")
    else:
        print("   ⚠️ Keine 'textclass' → textclass_count.csv übersprungen.")


def compute_categorical_counts(df_meta: pd.DataFrame, out_dir: Path, column: str, filename: str):
    """Helper: Zählt Werte in einer kategorialen Spalte."""
    if not has_column(df_meta, column):
        print(f"   ⚠️­️ Keine '{column}' → {filename} übersprungen.")
        return
    
    df_counts = (
        df_meta.groupby(column, dropna=True)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    out_path = out_dir / filename
    df_counts.to_csv(out_path, index=False, encoding="utf-8")
    print(f"   ✅ {filename}")


def compute_year_count_tokens(df_stop: pd.DataFrame, content_col: str, out_dir: Path):
    """Berechnet Token-Counts pro Jahr."""
    year_col = "year_final" if "year_final" in df_stop.columns else None
    
    if year_col is None:
        year_first, year = identify_year_columns(df_stop)
        if not year_first and not year:
            print("   ⚠️ Keine Jahr-Spalten → year_count_tokens.csv übersprungen.")
            return
        df_stop = coalesce_years(df_stop)
        year_col = "year_final"
    
    df = df_stop.copy()
    df["__tokens"] = df[content_col].astype(str).apply(count_tokens)
    
    df_year = (
        df.groupby(year_col, dropna=True)["__tokens"]
        .sum()
        .reset_index()
        .rename(columns={year_col: "year", "__tokens": "tokens"})
        .sort_values("year")
    )
    
    out_path = out_dir / "year_count_tokens.csv"
    df_year.to_csv(out_path, index=False, encoding="utf-8")
    print(f"   ✅ year_count_tokens.csv")


def compute_genre_per_source(df_meta: pd.DataFrame, out_dir: Path):
    """Erstellt Genre-pro-Source-Matrix."""
    if not (has_column(df_meta, "genre") and has_column(df_meta, "source")):
        print("   ⚠️ Keine 'genre' und/oder 'source' → genre_per_source.csv übersprungen.")
        return
    
    crosstab = pd.crosstab(df_meta["genre"], df_meta["source"])
    out_path = out_dir / "genre_per_source.csv"
    crosstab.to_csv(out_path, encoding="utf-8")
    print(f"   ✅ genre_per_source.csv")


def compute_tokens_per_author(df_stop: pd.DataFrame, content_col: str, out_dir: Path):
    """Berechnet Tokens pro Author."""
    if not has_column(df_stop, "author_surname"):
        print("   ⚠️ Keine 'author_surname' → tokens_per_author.csv übersprungen.")
        return
    
    df = df_stop.copy()
    df["__tokens"] = df[content_col].astype(str).apply(count_tokens)
    
    df_author = (
        df.groupby("author_surname", dropna=True)["__tokens"]
        .sum()
        .reset_index()
        .rename(columns={"__tokens": "tokens"})
        .sort_values("tokens", ascending=False)
    )
    
    out_path = out_dir / "tokens_per_author.csv"
    df_author.to_csv(out_path, index=False, encoding="utf-8")
    print(f"   ✅ tokens_per_author.csv")


def compute_tokens_per_genre(df_stop: pd.DataFrame, content_col: str, out_dir: Path):
    """Berechnet Tokens pro Genre."""
    if not has_column(df_stop, "genre"):
        print("   ⚠️ Keine 'genre' → tokens_per_genre.csv übersprungen.")
        return
    
    df = df_stop.copy()
    df["__tokens"] = df[content_col].astype(str).apply(count_tokens)
    
    df_genre = (
        df.groupby("genre", dropna=True)["__tokens"]
        .sum()
        .reset_index()
        .rename(columns={"__tokens": "tokens"})
        .sort_values("tokens", ascending=False)
    )
    
    out_path = out_dir / "tokens_per_genre.csv"
    df_genre.to_csv(out_path, index=False, encoding="utf-8")
    print(f"   ✅ tokens_per_genre.csv")


def compute_tokens_per_document(df_stop: pd.DataFrame, content_col: str, out_dir: Path):
    """Berechnet Tokens pro Dokument."""
    df = df_stop.copy()
    df["tokens"] = df[content_col].astype(str).apply(count_tokens)
    
    # Nur relevante Spalten behalten
    keep_cols = ["tokens"]
    if has_column(df, "author_surname"):
        keep_cols.append("author_surname")
    if has_column(df, "title"):
        keep_cols.append("title")
    
    # ID-Spalte hinzufügen (flexibel)
    id_col = identify_id_column(df)
    if id_col:
        keep_cols.insert(0, id_col)
    
    df_tokens = df[keep_cols].copy()
    
    out_path = out_dir / "tokens_per_document_stop.csv"
    df_tokens.to_csv(out_path, index=False, encoding="utf-8")
    print(f"   ✅ tokens_per_document_stop.csv")


def compute_rezensierte_autoren(df_meta: pd.DataFrame, out_dir: Path):
    """Extrahiert rezensierte Autoren aus Rezensions-Titeln."""
    if not (has_column(df_meta, "genre") and has_column(df_meta, "title")):
        print("   ⚠️ Keine 'genre' und/oder 'title' → rezensierte_autoren.csv übersprungen.")
        return
    
    df = df_meta.copy()
    df = df[df["genre"].astype(str).str.lower().str.contains("rezension", na=False)]
    
    if df.empty:
        print("   ⚠️­️ Keine Rezensionen gefunden → rezensierte_autoren.csv übersprungen.")
        return
    
    pattern = r"^([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)"
    df["reviewed_author"] = df["title"].astype(str).str.extract(pattern, expand=False)
    df_reviewed = df[df["reviewed_author"].notna()][["reviewed_author"]].copy()
    df_counts = df_reviewed["reviewed_author"].value_counts().reset_index()
    df_counts.columns = ["reviewed_author", "count"]
    
    out_path = out_dir / "rezensierte_autoren.csv"
    df_counts.to_csv(out_path, index=False, encoding="utf-8")
    print(f"   ✅ rezensierte_autoren.csv")


def compute_milestones(df_meta: pd.DataFrame, out_dir: Path):
    """Berechnet kumulative Milestones (Dokumente + Tokens pro Jahr)."""
    year_col = "year_final" if "year_final" in df_meta.columns else None
    
    if year_col is None:
        year_first, year = identify_year_columns(df_meta)
        if not year_first and not year:
            print("   ⚠️ Keine Jahr-Spalten → milestones.csv übersprungen.")
            return
        df_meta = coalesce_years(df_meta)
        year_col = "year_final"
    
    df_year = (
        df_meta.groupby(year_col, dropna=True)
        .size()
        .reset_index(name="documents")
        .sort_values(year_col)
    )
    df_year.columns = ["year", "documents"]
    df_year["cumulative_documents"] = df_year["documents"].cumsum()
    
    out_path = out_dir / "milestones.csv"
    df_year.to_csv(out_path, index=False, encoding="utf-8")
    print(f"   ✅ milestones.csv")


# =============================================================================
# run-Funktion für Pipeline
# =============================================================================

def run(
    preprocessed_dir: Path,
    output_dir: Path,
    delimiter: str = "auto",
) -> str:
    """
    Führt alle Statistik-Berechnungen durch.
    
    Returns:
        Verwendeter Delimiter
    """
    
    print(f"\n📁 Eingabeordner: {preprocessed_dir}")
    print(f"📁 Ausgabeordner: {output_dir}")
    print()
    
    ensure_nltk()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📚 Lade Korpora:")
    corpora, used_delimiter = load_corpus_files(preprocessed_dir, delimiter)
    
    # Metadaten von einem Korpus nehmen (alle sollten identische Metadaten haben)
    df_meta, _ = next(iter(corpora.values()))
    
    # stop-Variante für detaillierte Statistiken
    if "stop" in corpora:
        df_stop, content_col_stop = corpora["stop"]
    else:
        print("\n⚠️ korpus_stop.csv fehlt → einige Statistiken werden übersprungen.")
        df_stop, content_col_stop = None, None
    
    print("\n📊 Erstelle Statistiken:")
    
    # Statistiken
    compute_author_statistics(df_meta, output_dir)
    compute_token_statistics(corpora, output_dir)
    compute_textclass_and_documents(df_meta, output_dir)
    
    compute_categorical_counts(df_meta, output_dir, "address", "address.csv")
    compute_categorical_counts(df_meta, output_dir, "author_address", "author_address.csv")
    compute_categorical_counts(df_meta, output_dir, "source", "source.csv")
    compute_categorical_counts(df_meta, output_dir, "genre", "genre.csv")
    
    if df_stop is not None:
        compute_year_count_tokens(df_stop, content_col_stop, output_dir)
        compute_tokens_per_author(df_stop, content_col_stop, output_dir)
        compute_tokens_per_genre(df_stop, content_col_stop, output_dir)
        compute_tokens_per_document(df_stop, content_col_stop, output_dir)
    
    compute_genre_per_source(df_meta, output_dir)
    compute_rezensierte_autoren(df_meta, output_dir)
    compute_milestones(df_meta, output_dir)
    
    print("\n" + "="*60)
    print("✅ Alle Statistiken erstellt.")
    print("="*60)
    
    return used_delimiter


# =============================================================================
# Argumentparser
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erstellt Korpus-Statistiken aus preprocessed Dateien.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ÄNDERUNG v3:
  - Automatische Delimiter-Erkennung (--delimiter auto)
  - Automatische Content-Spalten-Erkennung
  - Flexible Jahr-/ID-Erkennung

Beispiel:
  python s01_3_statistics.py \\
      --preprocessed-dir output/processed_corpus \\
      --output-dir output/statistics \\
      --delimiter auto
        """
    )
    parser.add_argument(
        "--preprocessed-dir",
        required=True,
        type=Path,
        help="Ordner mit korpus_min/lem/stop.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Zielordner für Statistik-CSVs",
    )
    parser.add_argument(
        "--delimiter",
        default="auto",
        help="CSV-Delimiter ('auto' für automatische Erkennung, Standard: auto)",
    )
    return parser.parse_args(argv)


# =============================================================================
# Main (CLI-Wrapper)
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run(
        preprocessed_dir=args.preprocessed_dir,
        output_dir=args.output_dir,
        delimiter=args.delimiter,
    )


if __name__ == "__main__":
    main()
