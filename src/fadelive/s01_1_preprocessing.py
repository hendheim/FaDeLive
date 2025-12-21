#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocessing-Script für einen Korpus im CSV/TSV-Format.

ÄNDERUNG v3:
- Automatische Delimiter-Erkennung (Fallback: ";")
- Automatische Metadaten-Erkennung
- Flexible ID-Spalten-Erkennung
- Ausgabe verwendet erkannten Delimiter

Funktionen:
- Einlesen einer Datei mit Spalte `content` (und optionalen Metadaten)
- Drei Vorverarbeitungsstufen erzeugen:
    * min  : minimale Vorverarbeitung → content_min
    * lem  : Lemmatisierung → content_lem
    * stop : Lemmatisierung + Stoppwörterentfernung → content_stop
- Drei Ausgabedateien speichern:
    * korpus_min.csv (Metadaten + content_min)
    * korpus_lem.csv (Metadaten + content_lem)
    * korpus_stop.csv (Metadaten + content_stop)

Beispielaufruf:

    python s01_1_preprocessing.py \\
        --input data/raw/korpus.csv \\
        --output-dir output/processed_corpus \\
        --delimiter auto \\
        --replacements resources/replacements_v1.json \\
        --stopwords resources/stopwords_v1.txt \\
        --salat resources/ocr_post-correction_dictionary_v1.txt \\
        --hanta-model morphmodel_ger.pgz
"""

import argparse
import json
import re
import string
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional

import pandas as pd
from HanTa import HanoverTagger as ht

# Import der gemeinsamen Utils
try:
    from .pipeline_utils import (
        detect_delimiter, read_csv_auto,
        identify_content_column, identify_metadata_columns,
        identify_id_column, has_column, safe_filename
    )
except ImportError:
    from pipeline_utils import (
        detect_delimiter, read_csv_auto,
        identify_content_column, identify_metadata_columns,
        identify_id_column, has_column, safe_filename
    )


# ---------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------

def load_replacements(path: Path) -> Dict[str, str]:
    """Lädt eine JSON-Datei mit Ersetzungspaaren {pattern: replacement}."""
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_word_list(path: Path) -> Set[str]:
    """Lädt eine Wortliste (eine Form pro Zeile) als Set."""
    if not path or not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


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


EXTENDED_PUNCTUATION = string.punctuation + "»«â€ž§â€¹â€º—''â¸—â– "


def remove_punctuation(text: str) -> str:
    """Entfernt Interpunktion und spezielle Zeichen."""
    return text.translate(str.maketrans("", "", EXTENDED_PUNCTUATION))


def remove_words_by_list(text: str, removal_list: Set[str]) -> str:
    """Entfernt Tokens, die in removal_list enthalten sind."""
    tokens = re.findall(r"\b\w+\b[.,]?", text)
    cleaned = [
        tok for tok in tokens if tok.rstrip(".,").lower() not in removal_list
    ]
    return " ".join(cleaned)


def lemmatize_text(text: str, lemmatizer: ht.HanoverTagger) -> str:
    """Einfaches HanTa-based Lemmatisieren."""
    tokens = text.split()
    lemmas = [lemmatizer.analyze(tok)[0] for tok in tokens]
    return " ".join(lemmas)


# ---------------------------------------------------------
# Hauptvorverarbeitung
# ---------------------------------------------------------

def preprocess_text(
    text: str,
    *,
    replacements: Dict[str, str],
    stopwords: Set[str],
    salat: Set[str],
    lemmatizer: ht.HanoverTagger,
) -> Tuple[str, str, str]:
    """
    Erzeugt drei Vorverarbeitungsvarianten:
    - min  : minimale Vorverarbeitung
    - lem  : lemmatisierte Variante
    - stop : Lemma + Stopwortentfernung
    """
    if not isinstance(text, str) or not text.strip():
        return "", "", ""

    # --- MIN ---
    min_text = text.lower()
    min_text = apply_replacements(min_text, replacements)
    min_text = remove_words_by_list(min_text, salat)

    # --- LEM ---
    base = remove_punctuation(min_text)
    lem_text = lemmatize_text(base, lemmatizer)

    # --- STOP ---
    stop_text = remove_words_by_list(lem_text, stopwords)
    stop_text = remove_words_by_list(stop_text, salat)
    stop_text = apply_replacements(stop_text, replacements)

    return min_text, lem_text, stop_text


# ---------------------------------------------------------
# Speichern der Korpusvarianten
# ---------------------------------------------------------

def save_corpus_variants(
    df: pd.DataFrame, 
    out_dir: Path, 
    delimiter: str,
    original_metadata: List[str]
) -> None:
    """
    Speichert die drei Korpusvarianten in getrennte Dateien.
    
    Jede Datei enthält NUR die entsprechende verarbeitete Content-Spalte
    plus alle Original-Metadaten.
    """
    print(f"   📋 Metadaten-Spalten: {len(original_metadata)}")
    
    # MIN: Metadaten + content_min
    out_df_min = df[original_metadata + ["min"]].copy()
    out_df_min = out_df_min.rename(columns={"min": "content_min"})
    out_path_min = out_dir / "korpus_min.csv"
    out_df_min.to_csv(out_path_min, sep=delimiter, encoding="utf-8", index=False)
    print(f"   ✅ korpus_min.csv: {len(original_metadata)} Metadaten + content_min")
    
    # LEM: Metadaten + content_lem
    out_df_lem = df[original_metadata + ["lem"]].copy()
    out_df_lem = out_df_lem.rename(columns={"lem": "content_lem"})
    out_path_lem = out_dir / "korpus_lem.csv"
    out_df_lem.to_csv(out_path_lem, sep=delimiter, encoding="utf-8", index=False)
    print(f"   ✅ korpus_lem.csv: {len(original_metadata)} Metadaten + content_lem")
    
    # STOP: Metadaten + content_stop
    out_df_stop = df[original_metadata + ["stop"]].copy()
    out_df_stop = out_df_stop.rename(columns={"stop": "content_stop"})
    out_path_stop = out_dir / "korpus_stop.csv"
    out_df_stop.to_csv(out_path_stop, sep=delimiter, encoding="utf-8", index=False)
    print(f"   ✅ korpus_stop.csv: {len(original_metadata)} Metadaten + content_stop")


# ---------------------------------------------------------
# run-Funktion für Pipeline
# ---------------------------------------------------------

def run(
    input_path: Path,
    output_dir: Path,
    delimiter: str,
    replacements_path: Path,
    stopwords_path: Path,
    salat_path: Path,
    hanta_model: str,
) -> str:
    """
    Führt die komplette Preprocessing-Pipeline aus.
    
    Returns:
        Verwendeter Delimiter (für Weitergabe an nächste Schritte)
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 Lade Korpus: {input_path}")
    
    # Automatische Delimiter-Erkennung
    if delimiter == "auto":
        delimiter = detect_delimiter(input_path)
    
    df = pd.read_csv(input_path, sep=delimiter, encoding="utf-8")

    # Flexible Content-Spalten-Erkennung (content, text, body, etc.)
    content_col = identify_content_column(df)
    if content_col is None:
        raise ValueError(
            "Keine Content-Spalte gefunden. "
            "Erwartet eine Spalte namens: content, text, clean_text, body, fulltext"
        )
    
    # Falls nicht "content", umbenennen für konsistente Verarbeitung
    if content_col != "content":
        print(f"   ℹ️ Content-Spalte '{content_col}' wird als 'content' verwendet")
        df = df.rename(columns={content_col: "content"})

    print(f"   📊 {len(df)} Dokumente geladen")
    print(f"   🔍 Delimiter: {repr(delimiter)}")

    # Metadaten automatisch erkennen (vor Verarbeitung!)
    original_metadata = [col for col in df.columns if col != "content"]
    print(f"   📋 Erkannte Metadaten-Spalten: {len(original_metadata)}")
    
    # ID-Spalte identifizieren
    id_col = identify_id_column(df)
    if id_col:
        print(f"   🔑 ID-Spalte: {id_col}")

    print("\n📦 Lade Ressourcen ...")
    replacements = load_replacements(replacements_path)
    print(f"   ✔ Replacements: {len(replacements)} Regeln")
    
    stopwords = load_word_list(stopwords_path)
    print(f"   ✔ Stopwords: {len(stopwords)} Wörter")
    
    salat = load_word_list(salat_path)
    print(f"   ✔ OCR-Artefakte: {len(salat)} Einträge")
    
    lemmatizer = ht.HanoverTagger(hanta_model)
    print(f"   ✔ HanTa-Modell: {hanta_model}")

    print("\n📄 Starte Vorverarbeitung ...")
    min_list: List[str] = []
    lem_list: List[str] = []
    stop_list: List[str] = []

    for idx, text in enumerate(df["content"].astype(str), 1):
        if idx % 500 == 0:
            print(f"   Verarbeitet: {idx}/{len(df)} Dokumente", end="\r")
        
        min_t, lem_t, stop_t = preprocess_text(
            text,
            replacements=replacements,
            stopwords=stopwords,
            salat=salat,
            lemmatizer=lemmatizer,
        )
        min_list.append(min_t)
        lem_list.append(lem_t)
        stop_list.append(stop_t)

    print(f"   Verarbeitet: {len(df)}/{len(df)} Dokumente ✔")

    # Verarbeitete Spalten zum DataFrame hinzufügen
    df["min"] = min_list
    df["lem"] = lem_list
    df["stop"] = stop_list

    print("\n💾 Speichere Korpusvarianten ...")
    save_corpus_variants(df, output_dir, delimiter, original_metadata)

    print("\n" + "="*60)
    print("✅ Preprocessing erfolgreich abgeschlossen!")
    print("="*60)
    print(f"\n📁 Output-Verzeichnis: {output_dir}")
    print(f"   - korpus_min.csv ({len(original_metadata)} Metadaten + content_min)")
    print(f"   - korpus_lem.csv ({len(original_metadata)} Metadaten + content_lem)")
    print(f"   - korpus_stop.csv ({len(original_metadata)} Metadaten + content_stop)")
    print(f"   - Delimiter: {repr(delimiter)}")
    
    return delimiter


# ---------------------------------------------------------
# Argumentparser
# ---------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erzeugt drei Vorverarbeitungsvarianten eines Korpus aus CSV/TSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ÄNDERUNG v3:
  - Automatische Delimiter-Erkennung (--delimiter auto)
  - Automatische Metadaten-Erkennung
  - Flexible ID-Spalten-Erkennung

Beispiel:
  python s01_1_preprocessing.py \\
      --input data/raw/korpus.csv \\
      --output-dir output/processed_corpus \\
      --delimiter auto \\
      --replacements resources/replacements_v1.json \\
      --stopwords resources/stopwords_v1.txt \\
      --salat resources/ocr_post-correction_dictionary_v1.txt \\
      --hanta-model morphmodel_ger.pgz
        """
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Pfad zur Eingabedatei (CSV/TSV) mit Spalte 'content'.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Verzeichnis für Ausgabedateien.")
    parser.add_argument("--delimiter", default="auto",
                        help="Feldtrenner ('auto' für automatische Erkennung, Standard: auto).")
    parser.add_argument("--replacements", type=Path, required=True,
                        help="JSON-Datei mit Ersetzungspaaren.")
    parser.add_argument("--stopwords", type=Path, required=True,
                        help="Textdatei mit Stoppwörtern.")
    parser.add_argument("--salat", type=Path, required=True,
                        help="Liste mit OCR-Artefakten / Salatformen.")
    parser.add_argument("--hanta-model", type=str, default="morphmodel_ger.pgz",
                        help="Pfad zum HanTa-Modell.")
    return parser.parse_args(argv)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    run(
        input_path=args.input,
        output_dir=args.output_dir,
        delimiter=args.delimiter,
        replacements_path=args.replacements,
        stopwords_path=args.stopwords,
        salat_path=args.salat,
        hanta_model=args.hanta_model,
    )


if __name__ == "__main__":
    main()
