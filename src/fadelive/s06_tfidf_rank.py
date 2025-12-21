#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Erstellt Ranglisten von Dokumenten auf Basis aller TF-IDF-2000-Matrizen.

ÄNDERUNG v3:
- Verwendet gemeinsame pipeline_utils für konsistente Erkennung
- Erweiterte ID-Erkennung (doc_id > _id > id > filename)
- Content-Spalten-Ausschluss aus Features
- Konsistent mit Pipeline v3

Für alle CSV-Dateien in einem Eingabeordner (rekursiv), deren Dateiname
"tfidf" oder "dtm" enthält, wird:

    1) die TF-IDF/DTM-Matrix eingelesen,
    2) die relevanten Feature-Spalten identifiziert (automatisch),
    3) die Werte gefiltert (0 < tf-idf < 0.9),
    4) die wichtigsten Terme (Top-N) ermittelt,
    5) pro Dokument eine Summenkennzahl ("combined_sum") berechnet,
    6) eine Rangliste inkl. Metadaten erstellt,
    7) eine Vergleichsmatrix der Top-N-Terme ausgegeben.

Ausgaben (pro tfidf-Datei):

    <basisname>_doc_rank.csv   – Rangliste mit Metadaten
    <basisname>_vocab_rank.csv – TF-IDF-Werte der Top-N Terme (Term x Dokument)

Beispielaufruf:

    python s06_tfidf_rank.py \\
        --input-dir output \\
        --output-dir output/tfidf_rank \\
        --pattern "tfidf-2000" \\
        --top-n 2000
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Import der gemeinsamen Utils
try:
    from .pipeline_utils import (
        identify_content_column,
        identify_id_column
    )
except ImportError:
    from pipeline_utils import (
        identify_content_column,
        identify_id_column
    )


# =============================================================================
# FLEXIBLE METADATEN-ERKENNUNG
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
    "content", "text", "clean_text", "content_min", "content_lem", "content_stop", "content_gen"
}


def is_metadata_column(col_name: str) -> bool:
    """Prüft, ob eine Spalte eine Metadaten-Spalte ist."""
    col_lower = str(col_name).strip().lower()
    return col_name in KNOWN_METADATA_NAMES or col_lower in {n.lower() for n in KNOWN_METADATA_NAMES}


def identify_feature_columns(df: pd.DataFrame, exclude_content: bool = True) -> List[str]:
    """Identifiziert Feature-Spalten (= numerische Spalten, die KEINE Metadaten sind)."""
    feature_cols = []
    content_col = identify_content_column(df) if exclude_content else None
    
    for col in df.columns:
        if content_col and col == content_col:
            continue
        if is_metadata_column(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    
    return feature_cols


def identify_metadata_columns_rank(df: pd.DataFrame) -> List[str]:
    """Identifiziert Metadaten-Spalten (alle nicht-Feature-Spalten)."""
    feature_cols = set(identify_feature_columns(df, exclude_content=True))
    return [col for col in df.columns if col not in feature_cols]


# =============================================================================
# Hilfsfunktionen
# =============================================================================

def collect_tfidf_files(input_dir: Path, pattern: str = "tfidf") -> List[Path]:
    """Sammelt rekursiv alle CSV-Dateien, die das Pattern im Namen tragen."""
    files = []
    for root, _, filenames in os.walk(input_dir):
        for fname in filenames:
            if fname.endswith(".csv") and pattern in fname:
                files.append(Path(root) / fname)
    return files


def process_tfidf_file(
    tfidf_path: Path,
    output_dir: Path,
    top_n_terms: int = 2000,
) -> None:
    """Verarbeitet eine einzelne TF-IDF/DTM-Datei."""
    print(f"\n➡ Verarbeite Datei: {tfidf_path}")

    try:
        df = pd.read_csv(tfidf_path, encoding="utf-8")
    except Exception as e:
        print(f"   ⚠️ Fehler beim Einlesen, übersprungen: {e}")
        return

    # Metadaten und Features automatisch trennen
    meta_cols = identify_metadata_columns_rank(df)
    feature_cols = identify_feature_columns(df, exclude_content=True)
    
    if not feature_cols:
        print(f"   ⚠️ Keine Feature-Spalten erkannt – übersprungen")
        return
    
    content_col = identify_content_column(df)
    if content_col:
        print(f"   ⚠️ Warnung: Content-Spalte '{content_col}' in TF-IDF-Datei gefunden (wird ignoriert)")
    
    print(f"   📋 Erkannt: {len(meta_cols)} Metadaten, {len(feature_cols)} Features")

    # ID-Spalte finden
    id_col = identify_id_column(df)
    
    if id_col is None:
        print(f"   ⚠️ Keine ID-Spalte gefunden – übersprungen")
        return
    
    print(f"   🔑 ID-Spalte: {id_col}")
    
    df[id_col] = df[id_col].astype(str)

    # Features als numerisch erzwingen
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Filter: nur Werte im Bereich (0, 0.9)
    masked = df[feature_cols].where((df[feature_cols] > 0) & (df[feature_cols] < 0.9), 0.0)

    # Wichtigste Terme nach aufsummierter Stärke
    feature_summen = masked.sum(axis=0)

    top_n = min(top_n_terms, len(feature_summen))
    if top_n == 0:
        print(f"   ⚠️ Keine sinnvollen Werte in Datei – übersprungen")
        return

    top_terms = feature_summen.nlargest(top_n).index.tolist()
    print(f"   📊 Top-{top_n} Terme ausgewählt")

    # Summenbildung je Dokument über diese Top-Terme
    df["combined_sum"] = df[top_terms].sum(axis=1)

    # Rangbildung
    df_sorted = df.sort_values(by="combined_sum", ascending=False).reset_index(drop=True)
    df_sorted["rank"] = range(1, len(df_sorted) + 1)

    # Rangliste mit Metadaten
    available_meta_cols = [
        c for c in meta_cols 
        if c in df_sorted.columns and c != id_col and c != content_col
    ]
    rank_cols = [id_col] + available_meta_cols + ["combined_sum", "rank"]
    rank_with_meta = df_sorted[rank_cols].copy()
    
    # ID-Spalte standardisieren zu "id"
    if id_col != "id":
        rank_with_meta = rank_with_meta.rename(columns={id_col: "id"})
        print(f"   📄 ID-Spalte '{id_col}' → 'id'")

    # Vergleichsmatrix: Term x Dokument
    vergleich_df = df_sorted.set_index(id_col)[top_terms].transpose()
    vergleich_df.columns.name = None

    # Dateinamen ableiten
    basisname = tfidf_path.stem
    rang_full_pfad = output_dir / f"{basisname}_doc_rank.csv"
    vergleichspfad = output_dir / f"{basisname}_vocab_rank.csv"

    # Speichern
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_with_meta.to_csv(rang_full_pfad, index=False, encoding="utf-8")
    vergleich_df.to_csv(vergleichspfad, encoding="utf-8")

    print(f"   ✅ Rangliste: {rang_full_pfad.name} ({len(rank_with_meta)} Dokumente)")
    print(f"   ✅ Vergleich: {vergleichspfad.name} ({len(vergleich_df)} Terme)")


# =============================================================================
# run-Funktion für Pipeline
# =============================================================================

def run(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "tfidf",
    top_n: int = 2000,
) -> None:
    """Erzeugt für jede gefundene TF-IDF/DTM-CSV-Datei zwei Outputs."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    print(f"📁 Eingabeordner: {input_dir}")
    print(f"📁 Ausgabeordner: {output_dir}")
    print(f"🔽 Suchpattern: '{pattern}'")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n🔍 Suche TF-IDF/DTM-Dateien ...")
    files = collect_tfidf_files(input_dir, pattern)

    if not files:
        print(f"⚠️ Keine Dateien mit Pattern '{pattern}' gefunden.")
        return

    print(f"✅ {len(files)} Datei(en) gefunden.")

    for f in files:
        process_tfidf_file(
            tfidf_path=f,
            output_dir=output_dir,
            top_n_terms=top_n,
        )

    print("\n" + "="*60)
    print("✅ Verarbeitung abgeschlossen.")
    print("="*60)


# =============================================================================
# Argumente
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erzeugt Ranglisten aus TF-IDF/DTM-CSV-Dateien.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ÄNDERUNG v3:
  - Verwendet gemeinsame pipeline_utils
  - Erweiterte ID-Erkennung
  - Konsistent mit Pipeline v3

Beispiele:
  python s06_tfidf_rank.py --input-dir output --output-dir output/tfidf_rank --pattern "tfidf-2000"
        """
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Eingabeordner, in dem TF-IDF/DTM-CSV-Dateien gesucht werden (rekursiv).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Zielordner für Ranglisten & Vergleichsmatrizen.",
    )
    parser.add_argument(
        "--pattern",
        default="tfidf",
        help="Suchpattern für Dateinamen (Standard: 'tfidf').",
    )
    parser.add_argument(
        "--top-n",
        default=2000,
        type=int,
        help="Anzahl der wichtigsten Terme (Standard: 2000).",
    )
    return parser.parse_args(argv)


# =============================================================================
# Main (CLI-Wrapper)
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
