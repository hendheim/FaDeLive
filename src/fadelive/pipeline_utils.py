#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gemeinsame Utility-Funktionen für die NLP-Pipeline.

FEATURES:
- Automatische Delimiter-Erkennung (mit Fallback auf ";")
- Automatische Metadaten-Erkennung
- Flexible ID-Spalten-Erkennung
- Flexible Jahr-Spalten-Erkennung
- Konsistente Content-Spalten-Erkennung
"""

import csv
import re
from pathlib import Path
from typing import Optional, List, Tuple, Any

import pandas as pd


# =============================================================================
# AUTOMATISCHE DELIMITER-ERKENNUNG
# =============================================================================

def detect_delimiter(file_path: Path, sample_lines: int = 5) -> str:
    """
    Erkennt automatisch den Delimiter einer CSV/TSV-Datei.
    
    Prüft auf: ";", ",", "\t", "|"
    Fallback: ";"
    
    Args:
        file_path: Pfad zur Datei
        sample_lines: Anzahl der Zeilen zum Analysieren
    
    Returns:
        Erkannter Delimiter
    """
    if not file_path.exists():
        return ";"  # Fallback
    
    try:
        with file_path.open("r", encoding="utf-8") as f:
            sample = "".join([f.readline() for _ in range(sample_lines)])
        
        # csv.Sniffer verwenden
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t|")
            detected = dialect.delimiter
            print(f"   🔍 Delimiter erkannt: {repr(detected)}")
            return detected
        except csv.Error:
            pass
        
        # Fallback: Häufigkeitszählung
        delimiters = [";", ",", "\t", "|"]
        counts = {d: sample.count(d) for d in delimiters}
        
        # Nur Delimiter mit signifikanten Vorkommen berücksichtigen
        if max(counts.values()) > 0:
            best = max(counts, key=counts.get)
            if counts[best] > 2:  # Mindestens 3 Vorkommen
                print(f"   🔍 Delimiter erkannt (Heuristik): {repr(best)}")
                return best
        
    except Exception as e:
        print(f"   ⚠️  Delimiter-Erkennung fehlgeschlagen: {e}")
    
    print(f"   🔍 Delimiter-Fallback: ';'")
    return ";"


def read_csv_auto(file_path: Path, delimiter: Optional[str] = None, **kwargs) -> Tuple[pd.DataFrame, str]:
    """
    Liest eine CSV-Datei mit automatischer Delimiter-Erkennung.
    
    Args:
        file_path: Pfad zur Datei
        delimiter: Optionaler Delimiter (wenn None, wird automatisch erkannt)
        **kwargs: Weitere Argumente für pd.read_csv
    
    Returns:
        (DataFrame, verwendeter Delimiter)
    """
    if delimiter is None or delimiter == "auto":
        delimiter = detect_delimiter(file_path)
    
    df = pd.read_csv(file_path, sep=delimiter, encoding="utf-8", **kwargs)
    return df, delimiter


# =============================================================================
# CONTENT-SPALTEN-ERKENNUNG
# =============================================================================

# Prioritätsreihenfolge für Content-Spalten
CONTENT_COLUMN_PRIORITY = [
    "content_stop", "content_lem", "content_min", "content_gen",
    "content", "text", "clean_text", "body", "fulltext"
]


def identify_content_column(df: pd.DataFrame) -> Optional[str]:
    """
    Identifiziert die Content-Spalte flexibel.
    
    Prüft bekannte Namen in Prioritätsreihenfolge.
    
    Returns:
        Name der Content-Spalte oder None
    """
    lower_map = {str(c).lower(): c for c in df.columns}
    
    for cand in CONTENT_COLUMN_PRIORITY:
        if cand in df.columns:
            return cand
        lc = str(cand).lower()
        if lc in lower_map:
            return lower_map[lc]
    
    return None


def identify_content_column_strict(df: pd.DataFrame, expected: Optional[str] = None) -> str:
    """
    Wie identify_content_column, aber wirft Fehler wenn nicht gefunden.
    
    Args:
        df: DataFrame
        expected: Optional - erwarteter Spaltenname (z.B. "content_stop")
    
    Returns:
        Name der Content-Spalte
    
    Raises:
        ValueError: Wenn keine Content-Spalte gefunden
    """
    # Wenn erwartet, zuerst danach suchen
    if expected and expected in df.columns:
        return expected
    
    content_col = identify_content_column(df)
    if content_col is None:
        raise ValueError(
            f"Keine Content-Spalte gefunden. "
            f"Erwartet eine von: {', '.join(CONTENT_COLUMN_PRIORITY)}"
        )
    return content_col


# =============================================================================
# METADATEN-ERKENNUNG
# =============================================================================

# Bekannte Metadaten-Spaltennamen (zur Orientierung, nicht als Filter)
KNOWN_METADATA_NAMES = {
    # IDs
    "_id", "id", "doc_id", "document_id", "filename", "file_id",
    # Autor
    "author", "author_prename", "author_surname", "author_surname_norm", 
    "author_address", "author_address_geo",
    # Editor
    "editor_prename", "editor_surname",
    # Titel
    "title", "title_norm", "title_addition",
    # Quelle
    "source", "journal", "magazine", "publisher",
    # Zeit
    "year", "year_first", "year_final", "Jahr_final", "date", "jahr",
    # Ausgabe
    "volume", "edition", "issue", "pages", "pages_exzerpt",
    # Klassifikation
    "textclass", "genre", "category", "type",
    # Ort
    "address", "address_geo", "location", "place",
    # Sonstiges
    "lang", "language", "note", "archive", "female_education",
}


def identify_metadata_columns(df: pd.DataFrame, content_col: Optional[str] = None) -> List[str]:
    """
    Identifiziert alle Metadaten-Spalten (= alles außer Content).
    
    Args:
        df: DataFrame
        content_col: Optional - Name der Content-Spalte (wird automatisch erkannt wenn None)
    
    Returns:
        Liste der Metadaten-Spalten
    """
    if content_col is None:
        content_col = identify_content_column(df)
    
    # Alle Content-bezogenen Spalten ausschließen
    content_variants = {
        content_col, 
        "content", "content_min", "content_lem", "content_stop", "content_gen",
        "min", "lem", "stop", "gen",
        "text", "clean_text", "body", "fulltext"
    }
    
    return [col for col in df.columns if col not in content_variants]


# =============================================================================
# ID-SPALTEN-ERKENNUNG
# =============================================================================

# Prioritätsreihenfolge für ID-Spalten
ID_COLUMN_PRIORITY = [
    "_id", "id", "doc_id", "document_id", "filename", "file_id", "index"
]


def identify_id_column(df: pd.DataFrame) -> Optional[str]:
    """
    Identifiziert die ID-Spalte flexibel.
    
    Returns:
        Name der ID-Spalte oder None
    """
    lower_map = {str(c).lower(): c for c in df.columns}
    
    for cand in ID_COLUMN_PRIORITY:
        if cand in df.columns:
            return cand
        lc = str(cand).lower()
        if lc in lower_map:
            return lower_map[lc]
    
    return None


# =============================================================================
# JAHR-SPALTEN-ERKENNUNG
# =============================================================================

# Prioritätsreihenfolge für Jahr-Spalten
YEAR_FIRST_PRIORITY = ["year_first", "Jahr_first", "jahr_first"]
YEAR_PRIORITY = ["year", "jahr", "Jahr", "year_final", "Jahr_final", "date"]


def identify_year_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Identifiziert Jahr-Spalten flexibel.
    
    Returns:
        (year_first_column, year_column) - jeweils None wenn nicht gefunden
    """
    lower_map = {str(c).lower(): c for c in df.columns}
    
    year_first = None
    for cand in YEAR_FIRST_PRIORITY:
        if cand in df.columns:
            year_first = cand
            break
        lc = str(cand).lower()
        if lc in lower_map:
            year_first = lower_map[lc]
            break
    
    year = None
    for cand in YEAR_PRIORITY:
        if cand in df.columns:
            year = cand
            break
        lc = str(cand).lower()
        if lc in lower_map:
            year = lower_map[lc]
            break
    
    return year_first, year


def get_year_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Erstellt eine kombinierte Jahr-Serie (year_first hat Vorrang).
    
    Returns:
        Numerische Serie mit Jahren oder None
    """
    year_first_col, year_col = identify_year_columns(df)
    
    if not year_first_col and not year_col:
        return None
    
    to_num = lambda s: pd.to_numeric(s, errors="coerce")
    
    if year_first_col and year_first_col in df.columns:
        yf = to_num(df[year_first_col])
    else:
        yf = pd.Series(index=df.index, dtype="float64")
    
    if year_col and year_col in df.columns:
        y = to_num(df[year_col])
    else:
        y = pd.Series(index=df.index, dtype="float64")
    
    # year_first hat Vorrang
    return yf.where(~yf.isna(), y)


def coalesce_years(df: pd.DataFrame, output_col: str = "year_final") -> pd.DataFrame:
    """
    Erstellt eine kombinierte Jahr-Spalte (year_first hat Vorrang).
    
    Args:
        df: DataFrame
        output_col: Name der Ausgabespalte
    
    Returns:
        DataFrame mit zusätzlicher year_final Spalte
    """
    year_series = get_year_series(df)
    if year_series is not None:
        df = df.copy()
        df[output_col] = year_series
    return df


# =============================================================================
# INTERVALL-ERKENNUNG AUS DATEN
# =============================================================================

def detect_year_range(df: pd.DataFrame) -> Optional[Tuple[int, int]]:
    """
    Erkennt den Jahresbereich im Korpus.
    
    Returns:
        (min_year, max_year) oder None
    """
    year_series = get_year_series(df)
    if year_series is None or year_series.isna().all():
        return None
    
    min_year = int(year_series.min())
    max_year = int(year_series.max())
    
    return min_year, max_year


def generate_default_intervals(df: pd.DataFrame, interval_size: int = 15) -> List[Tuple[int, int]]:
    """
    Generiert Standard-Intervalle basierend auf dem Jahresbereich im Korpus.
    
    Args:
        df: DataFrame
        interval_size: Größe jedes Intervalls in Jahren
    
    Returns:
        Liste von (start, end) Tupeln
    """
    year_range = detect_year_range(df)
    if year_range is None:
        return []
    
    min_year, max_year = year_range
    intervals = []
    
    start = min_year
    while start <= max_year:
        end = min(start + interval_size - 1, max_year)
        intervals.append((start, end))
        start = end + 1
    
    return intervals


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def has_column(df: pd.DataFrame, col: str) -> bool:
    """Prüft, ob eine Spalte existiert und nicht-leere Werte enthält."""
    return col in df.columns and df[col].notna().any()


def safe_filename(name: str) -> str:
    """Erzeugt sichere Dateinamen."""
    # Ersetze problematische Zeichen
    result = str(name)
    for char in [" ", "/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        result = result.replace(char, "_")
    return result


def print_dataframe_info(df: pd.DataFrame, label: str = "DataFrame"):
    """Gibt Informationen über einen DataFrame aus."""
    content_col = identify_content_column(df)
    metadata_cols = identify_metadata_columns(df, content_col)
    id_col = identify_id_column(df)
    year_first, year = identify_year_columns(df)
    
    print(f"\n📊 {label}:")
    print(f"   Zeilen: {len(df)}")
    print(f"   Spalten: {len(df.columns)}")
    print(f"   Content-Spalte: {content_col or '—'}")
    print(f"   ID-Spalte: {id_col or '—'}")
    print(f"   Jahr-Spalten: year_first={year_first or '—'}, year={year or '—'}")
    print(f"   Metadaten: {len(metadata_cols)} Spalten")


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    # Einfacher Selbsttest
    print("Pipeline Utils - Selbsttest")
    print("=" * 40)
    
    # Test-DataFrame erstellen
    test_df = pd.DataFrame({
        "_id": [1, 2, 3],
        "title": ["A", "B", "C"],
        "year_first": [1800, 1850, 1900],
        "content_stop": ["text1", "text2", "text3"],
    })
    
    print_dataframe_info(test_df, "Test-DataFrame")
    
    print("\n✅ Selbsttest abgeschlossen.")
