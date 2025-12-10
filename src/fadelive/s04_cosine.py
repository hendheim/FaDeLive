#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Berechnung einer Cosinus-Ähnlichkeitsmatrix auf Grundlage von TF-IDF/DTM-Matrizen.

ÄNDERUNG v3 (Vollständige Flexibilisierung):
- Verwendet standardisierte Metadaten-Erkennungsfunktionen aus dem Pipeline-System
- Erweiterte ID-Erkennung (doc_id > _id > id > filename)
- Content-Spalten-Ausschluss (content_stop, content_lem, etc.)
- Konsistent mit anderen v2/v3/v4-Modulen
- Detaillierte Ausgabe über erkannte Spalten

Input-Datei:
    output/dtm_tfidf_stop/tfidf-2000.csv (oder jede andere TF-IDF/DTM-Matrix)

Output-Datei:
    output/cosine/cosine_tfidf2000.csv

Dieses Script:
    1) lädt die TF-IDF/DTM-Matrix
    2) trennt Metadaten von Feature-Spalten (automatisch, standardisiert)
    3) berechnet die Cosinus-Ähnlichkeitsmatrix
    4) speichert die Cosinus-Matrix als CSV

Beispielaufruf:

    python s04_cosine_v3.py \
        --input output/dtm_tfidf_stop/tfidf-2000.csv \
        --output output/cosine/cosine_tfidf2000.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity


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
    # Content-Spalten (werden NICHT als Features gezählt!)
    "content", "text", "clean_text", "content_min", "content_lem", "content_stop", "content_gen"
}


def identify_content_column(df: pd.DataFrame) -> Optional[str]:
    """
    Identifiziert die Content-Spalte flexibel.
    
    Priorität: content_gen > content_stop > content_lem > content_min > content > text > clean_text
    """
    candidates = [
        "content_gen", "content_stop", "content_lem", "content_min",
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


def identify_doc_id_column(df: pd.DataFrame) -> Optional[str]:
    """
    Identifiziert die Dokument-ID-Spalte flexibel.
    
    Priorität: doc_id > _id > id > filename
    """
    candidates = ["doc_id", "_id", "id", "filename"]
    lower_map = {str(c).lower(): c for c in df.columns}
    
    for cand in candidates:
        if cand in df.columns and df[cand].notna().any():
            return cand
        lc = str(cand).lower()
        if lc in lower_map and df[lower_map[lc]].notna().any():
            return lower_map[lc]
    
    return None


def is_metadata_column(col_name: str) -> bool:
    """Prüft, ob eine Spalte eine Metadaten-Spalte ist."""
    col_lower = str(col_name).strip().lower()
    if col_name in KNOWN_METADATA_NAMES or col_lower in {n.lower() for n in KNOWN_METADATA_NAMES}:
        return True
    return False


def identify_feature_columns(df: pd.DataFrame, exclude_content: bool = True) -> List[str]:
    """
    Identifiziert Feature-Spalten (= numerische Spalten, die KEINE Metadaten sind).
    
    Args:
        df: DataFrame
        exclude_content: Wenn True, werden Content-Spalten ausgeschlossen
    
    Returns:
        Liste der Feature-Spalten
    """
    feature_cols = []
    content_col = identify_content_column(df) if exclude_content else None
    
    for col in df.columns:
        # Content-Spalte ausschließen
        if content_col and col == content_col:
            continue
        # Bekannte Metadaten ausschließen
        if is_metadata_column(col):
            continue
        # Nur numerische Spalten
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    
    return feature_cols


def identify_metadata_columns(df: pd.DataFrame) -> List[str]:
    """
    Identifiziert Metadaten-Spalten (alle nicht-Feature-Spalten).
    
    Returns:
        Liste der Metadaten-Spalten
    """
    feature_cols = set(identify_feature_columns(df, exclude_content=True))
    metadata_cols = []
    
    for col in df.columns:
        if col not in feature_cols:
            metadata_cols.append(col)
    
    return metadata_cols


# =============================================================================
# Funktionen
# =============================================================================

def load_tfidf_matrix(path: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Lädt TF-IDF/DTM-Matrix und trennt Metadaten von Feature-Spalten (standardisiert).
    
    Returns:
        (df, metadata_columns, feature_columns)
    """
    if not path.exists():
        raise FileNotFoundError(f"❌ Datei nicht gefunden: {path}")

    print(f"📄 Lade Datei: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    
    # Automatische Trennung (standardisiert)
    meta_cols = identify_metadata_columns(df)
    feature_cols = identify_feature_columns(df, exclude_content=True)
    
    if not feature_cols:
        raise ValueError("❌ Keine Feature-Spalten gefunden.")
    
    # Content-Spalte erkennen (falls vorhanden, sollte aber nicht in TF-IDF sein)
    content_col = identify_content_column(df)
    if content_col:
        print(f"   ⚠️ Warnung: Content-Spalte '{content_col}' in Matrix gefunden (wird ignoriert)")
    
    # ID-Spalte erkennen
    id_col = identify_doc_id_column(df)
    if id_col:
        print(f"   🔑 ID-Spalte: {id_col}")
    
    print(f"   📋 Metadaten: {len(meta_cols)} Spalten")
    print(f"   📊 Features: {len(feature_cols)} Spalten")
    
    # Features bereinigen: numerisch erzwingen, NaN → 0
    features = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df[feature_cols] = features

    return df, meta_cols, feature_cols


def compute_cosine(df_features: pd.DataFrame) -> pd.DataFrame:
    """Berechnet die Cosinus-Ähnlichkeitsmatrix."""
    # Sicherheit: falls vorher doch irgendwo NaN geblieben ist
    if df_features.isna().values.any():
        raise ValueError("❌ Nach Bereinigung sind noch NaN in den Features vorhanden.")

    matrix = df_features.to_numpy(dtype=float)

    if matrix.size == 0:
        raise ValueError("❌ Die Matrix enthält keine Daten.")

    print(f"   🔢 Matrix-Größe: {matrix.shape[0]} Dokumente × {matrix.shape[1]} Features")
    
    cos = cosine_similarity(matrix)
    return pd.DataFrame(cos)


# =============================================================================
# run-Funktion für Pipeline
# =============================================================================

def run(
    input_path: Path,
    output_path: Path,
) -> None:
    """Berechnet eine Cosinusmatrix aus TF-IDF/DTM-CSV."""

    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    print()
    
    df, meta_cols, feature_cols = load_tfidf_matrix(input_path)

    # Dokument-IDs bestimmen (standardisiert)
    id_col = identify_doc_id_column(df)
    
    if id_col:
        doc_ids = df[id_col].fillna("").astype(str).tolist()
        print(f"   🆔 Verwende '{id_col}' als Dokument-ID")
    else:
        doc_ids = [f"doc_{i}" for i in range(len(df))]
        print(f"   🆔 Keine ID-Spalte gefunden, verwende generierte IDs (doc_0, doc_1, ...)")

    print("\n➡ Berechne Cosinus-Ähnlichkeit …")
    df_cos = compute_cosine(df[feature_cols])

    # Spalten und Index beschriften
    df_cos.index = doc_ids
    df_cos.columns = doc_ids

    # Zielordner anlegen
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Speichere Datei: {output_path}")
    df_cos.to_csv(output_path, encoding="utf-8", index=True)

    print(f"✅ Cosinusmatrix erfolgreich erstellt ({len(doc_ids)}×{len(doc_ids)})")


# =============================================================================
# Argumentparser
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Berechnet eine Cosinusmatrix aus TF-IDF/DTM-CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FLEXIBILISIERUNG v3:
  - Automatische Content-Spalten-Erkennung (werden ausgeschlossen)
  - Erweiterte ID-Erkennung (doc_id > _id > id > filename)
  - Flexible Feature-Erkennung (numerisch, keine Metadaten)
  - Konsistent mit anderen Pipeline-Modulen

Beispiele:
  # TF-IDF-Matrix
  python s04_cosine_v3.py \\
      --input output/dtm_tfidf_stop/tfidf-2000.csv \\
      --output output/cosine/cosine_tfidf2000.csv
  
  # DTM-Matrix
  python s04_cosine_v3.py \\
      --input output/dtm_tfidf_stop/dtm-2000.csv \\
      --output output/cosine/cosine_dtm2000.csv
        """
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Pfad zur TF-IDF/DTM-Eingabedatei.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Pfad zur Cosinus-Ausgabedatei.",
    )
    return parser.parse_args(argv)


# =============================================================================
# MAIN – CLI-Wrapper
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run(
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
