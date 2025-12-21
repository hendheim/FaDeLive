#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vorverarbeitung eines Korpus für Gensim-Modelle (Word2Vec, LDA, etc.).

ÄNDERUNG v3:
- Automatische Delimiter-Erkennung (Fallback: ";")
- Arbeitet mit neuen Content-Spaltennamen (content_min, content_lem, content_stop)
- Flexible Metadaten-Handhabung: Alle Spalten außer Content werden beibehalten
- Intervall-Unterstützung - Erzeugt separate Dateien pro Zeitintervall
- year_first hat Vorrang vor year bei Intervall-Filterung
- Ausgabe: Nur Metadaten + content_gen (KEINE anderen Content-Spalten)

Pipeline:
    1) Lowercasing
    2) Anwenden einer Ersetzungsliste (JSON)
    3) Normalisierung von Sonderzeichen:
         - alle Sonderzeichen werden zu Leerzeichen,
         - ., !, ? werden als eigene Tokens erhalten
    4) Entfernen von OCR-Artefakten ("Salat")
    5) HanTa-Lemmatisierung (., !, ? bleiben als eigene Tokens erhalten)
    6) Entfernen von Stopwörtern (., !, ? bleiben erhalten)
    7) (optional) Entfernen von ., !, ? aus dem finalen Text

Input:
    korpus_min.csv (mit content_min)

Output:
    - korpus_gen.csv (Gesamtkorpus, Metadaten + content_gen)
    - korpus_gen_<interval>.csv (pro Intervall, nur Metadaten + content_gen)

Beispielaufruf:

    python s02_preprocessing_gensim.py \\
        --input output/processed_corpus/korpus_min.csv \\
        --output output/processed_corpus/korpus_gen.csv \\
        --delimiter auto \\
        --replacements resources/replacements_v1.json \\
        --stopwords resources/stopwords_v1.txt \\
        --salat resources/ocr_post-correction_dictionary.txt \\
        --hanta-model resources/morphmodel_ger.pgz
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
from HanTa import HanoverTagger as ht

# Import der gemeinsamen Utils
try:
    from .pipeline_utils import (
        detect_delimiter,
        identify_content_column,
        identify_metadata_columns,
        identify_year_columns,
        get_year_series,
        has_column
    )
except ImportError:
    from pipeline_utils import (
        detect_delimiter,
        identify_content_column,
        identify_metadata_columns,
        identify_year_columns,
        get_year_series,
        has_column
    )


# ---------------------------------------------------------
# Konfiguration / Parameter
# ---------------------------------------------------------

# Satzzeichen, die explizit als eigene Tokens erhalten werden sollen
ALLOWED_PUNCT = {".", "!", "?"}

# Standardwerte (werden vom CLI überschrieben)
DEFAULT_DELIMITER = "auto"
DEFAULT_HANTA_MODEL = "morphmodel_ger.pgz"


# ---------------------------------------------------------
# Intervall-Verarbeitung
# ---------------------------------------------------------

def parse_interval(interval_str: str) -> Tuple[int, int]:
    """
    Parst einen Intervall-String wie "1784-1796" in (start, end).
    
    Raises:
        ValueError: Bei ungültigem Format
    """
    parts = interval_str.split("-")
    if len(parts) != 2:
        raise ValueError(f"Ungültiges Intervall-Format: {interval_str}. Erwartet: 'YYYY-YYYY'")
    
    try:
        start = int(parts[0].strip())
        end = int(parts[1].strip())
    except ValueError:
        raise ValueError(f"Intervall enthält ungültige Zahlen: {interval_str}")
    
    if start > end:
        raise ValueError(f"Start-Jahr ({start}) ist größer als End-Jahr ({end})")
    
    return start, end


def filter_by_interval(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Filtert DataFrame nach Zeitintervall.
    year_first hat Vorrang vor year.
    
    Returns:
        Gefiltertes DataFrame oder leeres DataFrame wenn keine Jahr-Spalten vorhanden
    """
    year_series = get_year_series(df)
    
    if year_series is None:
        print(f"   ⚠️ Keine Jahr-Spalten vorhanden, Intervall-Filterung übersprungen")
        return pd.DataFrame()
    
    mask = (year_series >= start_year) & (year_series <= end_year)
    filtered = df[mask].copy()
    
    print(f"   📅 Intervall {start_year}-{end_year}: {len(filtered)} von {len(df)} Dokumenten gefunden")
    
    return filtered


# ---------------------------------------------------------
# Ressourcen laden
# ---------------------------------------------------------

def load_list(path: Path | None) -> set:
    """Lädt eine Wortliste (Stopwörter, OCR-Salat) als Set."""
    if path is None or not path.exists():
        return set()
    try:
        with path.open("r", encoding="utf-8") as f:
            return {line.strip().lower() for line in f if line.strip()}
    except FileNotFoundError:
        print(f"⚠️  Warnung: Liste nicht gefunden: {path}")
        return set()


def load_replacements(path: Path | None) -> dict:
    """Lädt eine JSON-Ersetzungsliste."""
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Warnung: Ersetzungsdatei nicht gefunden: {path}")
        return {}


# ---------------------------------------------------------
# Token-/Text-Level Funktionen
# ---------------------------------------------------------

def apply_replacements(text: str, replacements: dict) -> str:
    """Wendet String- und Regex-Ersetzungen an."""
    
    def is_regex(pattern: str) -> bool:
        regex_indicators = [
            r'\(\?', r'\[.+\]', r'\\b', r'\\B', r'\\d', r'\\w', r'\\s',
            r'[^\\][\*\+\?]', r'\{\d+', r'^\^', r'\$$', r'[^\\]\|'
        ]
        for indicator in regex_indicators:
            if re.search(indicator, pattern):
                return True
        return False
    
    for pattern, replacement in replacements.items():
        if is_regex(pattern):
            try:
                text = re.sub(pattern, replacement, text)
            except re.error as e:
                print(f"⚠️  Regex-Fehler: '{pattern}' - {e}")
                continue
        else:
            text = text.replace(pattern, replacement)
    
    return text


def normalize_punctuation(text: str, keep: set[str] = ALLOWED_PUNCT) -> str:
    """
    Normalisiert Sonderzeichen:
      - alphanumerische Zeichen bleiben
      - Satzzeichen in `keep` werden als eigene Tokens ausgegeben
      - alle anderen Zeichen werden zu Leerzeichen
    """
    out = []
    for ch in text:
        if ch in keep:
            out.append(f" {ch} ")
        elif ch.isalnum():
            out.append(ch)
        else:
            out.append(" ")

    text = "".join(out)
    return re.sub(r"\s+", " ", text).strip()


def remove_salat(text: str, salat: set) -> str:
    """Entfernt bekannte OCR-Artefakte (exakte Token-Treffer)."""
    return " ".join(t for t in text.split() if t.lower() not in salat)


def lemmatize(text: str, tagger: ht.HanoverTagger) -> str:
    """
    Lemmatisiert mit HanTa.
    - Tokens, die genau ., ! oder ? sind, werden unverändert übernommen.
    - alle anderen Tokens werden mit HanTa lemmatisiert.
    """
    out = []
    for token in text.split():
        if token in ALLOWED_PUNCT:
            out.append(token)
            continue
        lemma = tagger.analyze(token)[0].split("|")[0]
        out.append(lemma)
    return " ".join(out)


def remove_stopwords(text: str, stopwords: set) -> str:
    """Entfernt Stopwörter; ., !, ? bleiben als Tokens stehen."""
    cleaned = []
    for token in text.split():
        if token in ALLOWED_PUNCT:
            cleaned.append(token)
        elif token.lower() not in stopwords:
            cleaned.append(token)
    return " ".join(cleaned)


def remove_sentence_punct(text: str, keep: set[str] | None = None) -> str:
    """Entfernt Satzzeichen (. ! ?) als eigene Tokens."""
    if keep is None:
        keep = set()

    out = []
    for tok in text.split():
        if tok in ALLOWED_PUNCT and tok not in keep:
            continue
        out.append(tok)
    return " ".join(out)


# ---------------------------------------------------------
# Vollständige Pipeline
# ---------------------------------------------------------

def process_text(
    text: str,
    *,
    replacements: dict,
    stopwords: set,
    salat: set,
    lemmatizer: ht.HanoverTagger,
    keep_sentence_punct: bool = True,
) -> str:

    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = apply_replacements(text, replacements)
    text = normalize_punctuation(text)
    text = remove_salat(text, salat)
    text = lemmatize(text, lemmatizer)
    text = remove_stopwords(text, stopwords)

    if not keep_sentence_punct:
        text = remove_sentence_punct(text)

    return text.strip()


# ---------------------------------------------------------
# run-Funktion
# ---------------------------------------------------------

def run(
    input_path: Path,
    output_path: Path,
    delimiter: str = "auto",
    replacements_path: Optional[Path] = None,
    stopwords_path: Optional[Path] = None,
    salat_path: Optional[Path] = None,
    hanta_model: str = DEFAULT_HANTA_MODEL,
    keep_sentence_punct: bool = True,
    intervals: Optional[List[str]] = None,
) -> str:
    """
    Hauptfunktion: Lädt Korpus, verarbeitet, speichert Ausgaben.
    
    Returns:
        Verwendeter Delimiter
    """

    # Delimiter erkennen
    if delimiter == "auto":
        delimiter = detect_delimiter(input_path)

    # 1) Korpus laden
    print(f"📄 Lade Korpus: {input_path}")
    df = pd.read_csv(input_path, sep=delimiter, encoding="utf-8")

    # Content-Spalte identifizieren
    content_col = identify_content_column(df)
    if content_col is None:
        raise ValueError("Keine Content-Spalte gefunden")
    print(f"📋 Erkannte Content-Spalte: {content_col}")

    # Metadaten identifizieren
    metadata_cols = identify_metadata_columns(df, content_col)
    print(f"📋 Erkannte Metadaten: {len(metadata_cols)} Spalten")

    # 2) Ressourcen laden
    print("📦 Lade Ressourcen ...")
    replacements = load_replacements(replacements_path)
    stopwords = load_list(stopwords_path)
    salat = load_list(salat_path)
    lemmatizer = ht.HanoverTagger(hanta_model)

    print(f"   ✔ {len(replacements)} Ersetzungen")
    print(f"   ✔ {len(stopwords)} Stopwörter")
    print(f"   ✔ {len(salat)} OCR-Artefakte")

    # 3) Verarbeitung
    print(f"\n📄 Verarbeite {len(df)} Dokumente ...")
    
    processed_texts = []
    for idx, text in enumerate(df[content_col].astype(str), 1):
        if idx % 100 == 0:
            print(f"   {idx}/{len(df)} ...", end="\r")
        
        proc = process_text(
            text,
            replacements=replacements,
            stopwords=stopwords,
            salat=salat,
            lemmatizer=lemmatizer,
            keep_sentence_punct=keep_sentence_punct,
        )
        processed_texts.append(proc)
    
    print(f"   ✔ {len(df)} Dokumente verarbeitet")

    # 4) Gesamtkorpus speichern (Metadaten + content_gen)
    print(f"\n💾 Speichere Gesamtkorpus: {output_path}")
    df_out = df[metadata_cols].copy()
    df_out["content_gen"] = processed_texts
    df_out.to_csv(output_path, sep=delimiter, index=False, encoding="utf-8")
    print(f"   ✔ Gespeichert: {len(df_out)} Dokumente")

    # 5) Intervalle (optional)
    if intervals:
        print(f"\n📅 Verarbeite {len(intervals)} Intervalle ...")
        
        output_dir = output_path.parent
        output_stem = output_path.stem
        
        for interval_str in intervals:
            try:
                start_year, end_year = parse_interval(interval_str)
            except ValueError as e:
                print(f"   ⚠️  Überspringe ungültiges Intervall '{interval_str}': {e}")
                continue
            
            df_filtered = filter_by_interval(df, start_year, end_year)
            
            if df_filtered.empty:
                print(f"   ⚠️  Keine Dokumente für {interval_str} → übersprungen")
                continue
            
            # Nur gefilterte Dokumente verwenden
            filtered_indices = df_filtered.index
            filtered_processed = [processed_texts[i] for i in filtered_indices]
            
            df_interval = df_filtered[metadata_cols].copy()
            df_interval["content_gen"] = filtered_processed
            
            interval_file = output_dir / f"{output_stem}_{interval_str}.csv"
            df_interval.to_csv(interval_file, sep=delimiter, index=False, encoding="utf-8")
            print(f"   ✔ Gespeichert: {interval_file.name} ({len(df_interval)} Dokumente)")

    print("\n✅ Verarbeitung abgeschlossen.")
    return delimiter


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gensim-Preprocessing eines Korpus (unterstützt Intervalle)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ÄNDERUNG v3:
  - Automatische Delimiter-Erkennung (--delimiter auto)
  - Flexible Content-/Metadaten-Erkennung
  - Konsistent mit Pipeline v3

Beispiel:
  python s02_preprocessing_gensim.py \\
      --input output/processed_corpus/korpus_min.csv \\
      --output output/processed_corpus/korpus_gen.csv \\
      --delimiter auto
        """
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Pfad zur Eingabedatei (korpus_min.csv mit content_min)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Pfad zur Ausgabedatei (korpus_gen.csv)",
    )
    parser.add_argument(
        "--delimiter",
        default=DEFAULT_DELIMITER,
        help="CSV-Delimiter ('auto' für automatische Erkennung)",
    )
    parser.add_argument(
        "--replacements",
        type=Path,
        help="JSON-Datei mit Ersetzungspaaren",
    )
    parser.add_argument(
        "--stopwords",
        type=Path,
        help="Textdatei mit Stopwörtern",
    )
    parser.add_argument(
        "--salat",
        type=Path,
        help="Textdatei mit OCR-Artefakten",
    )
    parser.add_argument(
        "--hanta-model",
        default=DEFAULT_HANTA_MODEL,
        help=f"Pfad zum HanTa-Modell (Standard: {DEFAULT_HANTA_MODEL})",
    )
    parser.add_argument(
        "--remove-sentence-punct",
        action="store_true",
        help="Entfernt . ! ? aus dem Ergebnis",
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        help='Zeitintervalle (z.B. "1784-1796" "1797-1810")',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    run(
        input_path=args.input,
        output_path=args.output,
        delimiter=args.delimiter,
        replacements_path=args.replacements,
        stopwords_path=args.stopwords,
        salat_path=args.salat,
        hanta_model=args.hanta_model,
        keep_sentence_punct=not args.remove_sentence_punct,
        intervals=args.intervals,
    )


if __name__ == "__main__":
    main()
