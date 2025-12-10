#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistik-Pipeline für den Korpus.

ÄNDERUNG v3 (Vollständige Flexibilisierung):
- Automatische Content-Spalten-Erkennung (content_min, content_lem, content_stop, content)
- Verwendet standardisierte Metadaten-Erkennungsfunktionen
- Flexible Jahr-Erkennung (year_first, year, Jahr_final, jahr)
- Flexible ID-Erkennung (doc_id, _id, id, filename)
- Konsistent mit Pipeline v2-Outputs

**WICHTIG:** Pipeline v2-Änderung bei Content-Spalten:
- korpus_min.csv enthält: Metadaten + content_min (NICHT "content"!)
- korpus_lem.csv enthält: Metadaten + content_lem
- korpus_stop.csv enthält: Metadaten + content_stop

Eingabe:
    output/processed_corpus/korpus_min.csv (mit content_min)
    output/processed_corpus/korpus_lem.csv (mit content_lem)
    output/processed_corpus/korpus_stop.csv (mit content_stop)

Ausgabe:
    CSV-Dateien in output/statistics/, je nach vorhandenen Metadaten:
        author_statistics.csv (falls author_surname vorhanden)
        tokens.csv
        tokens_per_textclass.csv (falls textclass vorhanden)
        textclass_count.csv (falls textclass vorhanden)
        documents_count.csv
        address.csv (falls address vorhanden)
        author_address.csv (falls author_address vorhanden)
        source.csv (falls source vorhanden)
        genre.csv (falls genre vorhanden)
        year_count_tokens.csv (falls year/year_first vorhanden)
        genre_per_source.csv (falls beide vorhanden)
        tokens_per_author.csv (falls author_surname vorhanden)
        tokens_per_genre.csv (falls genre vorhanden)
        tokens_per_document_stop.csv
        rezensierte_autoren.csv (falls genre und title vorhanden)
        milestones.csv (falls year/year_first vorhanden)

Beispielaufruf:

    python s01_3_statistics_v3.py \
        --preprocessed-dir output/processed_corpus \
        --output-dir output/statistics \
        --delimiter ";"
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


# =============================================================================
# FLEXIBLE METADATEN-ERKENNUNG (standardisiert)
# =============================================================================

KNOWN_METADATA_NAMES = {
    "_id", "id", "doc_id", "filename",
    "author", "author_prename", "author_surname", "author_surname_norm", "author_address", "author_address_geo",
    "editor_prename", "editor_surname",
    "title", "title_norm", "title_addition",
    "source", "journal", "magazine",
    "year", "year_first", "year_final", "Jahr_final",
    "volume", "edition", "issue", "pages", "pages_exzerpt",
    "textclass", "genre", "address", "address_geo",
    "lang", "language", "note", "archive",
    "female_education",
}


def identify_content_column(df: pd.DataFrame) -> Optional[str]:
    """
    Identifiziert die Content-Spalte flexibel.
    
    Priorität: content_stop > content_lem > content_min > content_gen > content > text > clean_text
    """
    candidates = [
        "content_stop", "content_lem", "content_min", "content_gen",
        "content", "text", "clean_text"
    ]
    lower_map = {str(c).lower(): c for c in df.columns}
    
    for cand in candidates:
        if cand in df.columns:
            return cand
        lc = str(cand).lower()
        if lc in lower_map:
            return lower_map[lc]
    
    return None


def identify_year_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Identifiziert Jahr-Spalten flexibel.
    
    Returns:
        (year_first_column, year_column)
    """
    year_first_candidates = ["year_first", "Jahr_first"]
    year_candidates = ["year", "jahr", "Jahr", "year_final", "Jahr_final"]
    
    lower_map = {str(c).lower(): c for c in df.columns}
    
    year_first = None
    for cand in year_first_candidates:
        if cand in df.columns:
            year_first = cand
            break
        lc = str(cand).lower()
        if lc in lower_map:
            year_first = lower_map[lc]
            break
    
    year = None
    for cand in year_candidates:
        if cand in df.columns:
            year = cand
            break
        lc = str(cand).lower()
        if lc in lower_map:
            year = lower_map[lc]
            break
    
    return year_first, year


def coalesce_years(df: pd.DataFrame) -> pd.DataFrame:
    """Erstellt year_final aus year_first/year (flexibel)."""
    year_first_col, year_col = identify_year_columns(df)
    to_num = lambda s: pd.to_numeric(s, errors="coerce")
    yf = to_num(df[year_first_col]) if year_first_col and year_first_col in df.columns else pd.Series(index=df.index, dtype="float64")
    y  = to_num(df[year_col]) if year_col and year_col in df.columns else pd.Series(index=df.index, dtype="float64")
    df["year_final"] = yf.where(~yf.isna(), y)
    return df


def identify_metadata_columns(df: pd.DataFrame) -> List[str]:
    """
    Identifiziert alle Metadaten-Spalten (= alles außer Content).
    
    Returns:
        Liste der Metadaten-Spalten
    """
    content_col = identify_content_column(df)
    return [col for col in df.columns if col != content_col]


def has_column(df: pd.DataFrame, col: str) -> bool:
    """Prüft, ob eine Spalte existiert und nicht-leere Werte enthält."""
    return col in df.columns and df[col].notna().any()


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
# Laden der Korpora (FLEXIBILISIERT)
# =============================================================================

def load_corpus_files(preprocessed_dir: Path, delimiter: str = "\t") -> dict:
    """
    Lädt korpus_min/lem/stop.csv, falls vorhanden.
    
    **WICHTIG:** Erkennt automatisch Content-Spalten:
    - korpus_min.csv → content_min
    - korpus_lem.csv → content_lem
    - korpus_stop.csv → content_stop

    Returns:
        dict: {"min": (df_min, content_col), "lem": (df_lem, content_col), "stop": (df_stop, content_col)}
    """
    corpora = {}
    for variant in ("min", "lem", "stop"):
        path = preprocessed_dir / f"korpus_{variant}.csv"
        if path.exists():
            print(f"   📄 Lade {path.name}")
            df = pd.read_csv(path, sep=delimiter, encoding="utf-8")
            
            # Content-Spalte automatisch erkennen
            content_col = identify_content_column(df)
            if content_col is None:
                print(f"      ⚠️ Keine Content-Spalte gefunden in {path.name}, übersprungen.")
                continue
            
            print(f"      ✓ Content-Spalte: {content_col}")
            
            # Jahr-Spalten zusammenführen (falls vorhanden)
            year_first, year = identify_year_columns(df)
            if year_first or year:
                df = coalesce_years(df)
                print(f"      ✓ Jahr-Spalten: year_first={year_first or '—'}, year={year or '—'} → year_final")
            
            corpora[variant] = (df, content_col)
        else:
            print(f"   ⚠️ {path.name} nicht gefunden, übersprungen.")
    
    if not corpora:
        raise FileNotFoundError(f"Keine korpus_*.csv in {preprocessed_dir} gefunden.")
    
    return corpora


# =============================================================================
# Statistik-Funktionen (ANGEPASST für flexible Content-Spalten)
# =============================================================================

def compute_author_statistics(df_meta: pd.DataFrame, out_dir: Path):
    """Erstellt Author-Statistiken (falls author_surname vorhanden)."""
    if not has_column(df_meta, "author_surname"):
        print("   ⏭️ Keine 'author_surname' → author_statistics.csv übersprungen.")
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
    """
    Berechnet Token-Statistiken über alle Varianten.
    
    **WICHTIG:** Verwendet die erkannte Content-Spalte für jede Variante!
    """
    tokens_rows = []
    tokens_per_tc_rows = []

    for variant, (df, content_col) in corpora.items():
        variant_name = variant  # "min", "lem", "stop"
        df = df.copy()

        # Token zählen aus der FLEXIBLEN Content-Spalte!
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
        print("   ⏭️ Keine 'textclass' → tokens_per_textclass.csv übersprungen.")


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
        print("   ⏭️ Keine 'textclass' → textclass_count.csv übersprungen.")


def compute_categorical_counts(df_meta: pd.DataFrame, out_dir: Path, column: str, filename: str):
    """Helper: Zählt Werte in einer kategorialen Spalte."""
    if not has_column(df_meta, column):
        print(f"   ⏭️ Keine '{column}' → {filename} übersprungen.")
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
    """
    Berechnet Token-Counts pro Jahr.
    
    **WICHTIG:** Verwendet flexible Jahr-Erkennung (year_final aus year_first/year).
    """
    # Jahr-Spalte verwenden (sollte bereits year_final sein)
    year_col = "year_final" if "year_final" in df_stop.columns else None
    
    if year_col is None:
        year_first, year = identify_year_columns(df_stop)
        if not year_first and not year:
            print("   ⏭️ Keine Jahr-Spalten → year_count_tokens.csv übersprungen.")
            return
        # Falls year_final nicht existiert, erstellen
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
        print("   ⏭️ Keine 'genre' und/oder 'source' → genre_per_source.csv übersprungen.")
        return
    
    crosstab = pd.crosstab(df_meta["genre"], df_meta["source"])
    out_path = out_dir / "genre_per_source.csv"
    crosstab.to_csv(out_path, encoding="utf-8")
    print(f"   ✅ genre_per_source.csv")


def compute_tokens_per_author(df_stop: pd.DataFrame, content_col: str, out_dir: Path):
    """Berechnet Tokens pro Author."""
    if not has_column(df_stop, "author_surname"):
        print("   ⏭️ Keine 'author_surname' → tokens_per_author.csv übersprungen.")
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
        print("   ⏭️ Keine 'genre' → tokens_per_genre.csv übersprungen.")
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
    id_col = None
    for cand in ["doc_id", "_id", "id", "filename"]:
        if has_column(df, cand):
            id_col = cand
            keep_cols.insert(0, id_col)
            break
    
    df_tokens = df[keep_cols].copy()
    
    out_path = out_dir / "tokens_per_document_stop.csv"
    df_tokens.to_csv(out_path, index=False, encoding="utf-8")
    print(f"   ✅ tokens_per_document_stop.csv")


def compute_rezensierte_autoren(df_meta: pd.DataFrame, out_dir: Path):
    """Extrahiert rezensierte Autoren aus Rezensions-Titeln."""
    if not (has_column(df_meta, "genre") and has_column(df_meta, "title")):
        print("   ⏭️ Keine 'genre' und/oder 'title' → rezensierte_autoren.csv übersprungen.")
        return
    
    df = df_meta.copy()
    df = df[df["genre"].astype(str).str.lower().str.contains("rezension", na=False)]
    
    if df.empty:
        print("   ⏭️ Keine Rezensionen gefunden → rezensierte_autoren.csv übersprungen.")
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
            print("   ⏭️ Keine Jahr-Spalten → milestones.csv übersprungen.")
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
    delimiter: str = ";",
) -> None:
    """Führt alle Statistik-Berechnungen durch."""
    
    print(f"\n📁 Eingabeordner: {preprocessed_dir}")
    print(f"📁 Ausgabeordner: {output_dir}")
    print()
    
    ensure_nltk()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📂 Lade Korpora:")
    corpora = load_corpus_files(preprocessed_dir, delimiter)
    
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


# =============================================================================
# Argumentparser
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erstellt Korpus-Statistiken aus preprocessed Dateien.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FLEXIBILISIERUNG v3:
  - Automatische Content-Spalten-Erkennung (content_min, content_lem, content_stop)
  - Flexible Jahr-Erkennung (year_first, year, Jahr_final, jahr)
  - Flexible ID-Erkennung (doc_id, _id, id, filename)
  - Konsistent mit Pipeline v2-Outputs

Beispiel:
  python s01_3_statistics_v3.py \\
      --preprocessed-dir output/processed_corpus \\
      --output-dir output/statistics \\
      --delimiter ";"
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
        default=";",
        help="CSV-Delimiter (Standard: ';')",
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
