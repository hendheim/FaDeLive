#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NLP-Pipeline v3 - Mit automatischer Erkennung

FEATURES:
- Automatische Delimiter-Erkennung (Fallback: ";")
- Automatische Metadaten-Erkennung
- Flexible ID-Spalten-Erkennung  
- Dynamische Intervall-Generierung
- Kohärente Parameterübergabe zwischen Schritten

Pipeline-Schritte:
    1. s01_1_preprocessing: Vorverarbeitung → korpus_min/lem/stop.csv
    2. s01_2_vocabular: Vokabular-Erzeugung (mit dynamischen Intervallen)
    3. s01_3_statistics: Statistik-Erzeugung
    4. s01_4_pos_tag: POS-Tagging der Top-5000 Ausdrücke
    5. s02_preprocessing_gensim: Gensim-Preprocessing → korpus_gen.csv
    6. s03_dtm_tfidf: DTM- und TF-IDF-Matrizen
    7. s04_cosine: Kosinus-Matrizen
    8. s05_dtm_tfidf_cos_intervals: Intervall-Matrizen
    9. s06_tfidf_rank: TF-IDF-Ranglisten
   10. s07_word_vector_model: Word2Vec-Modelle
"""

from pathlib import Path
import tomllib
import argparse
from typing import Optional

# Import der Module (mit Fallback für Standalone-Ausführung)
try:
    from .s01_1_preprocessing import run as step1
    from .s01_2_vocabular import run as step2
    from .s01_3_statistics import run as step3
    from .s01_4_pos_tag import run as step4
    from .s02_preprocessing_gensim import run as step5
    from .s03_dtm_tfidf import run as step6
    from .s04_cosine import run as step7
    from .s05_dtm_tfidf_cos_intervals import run as step8
    from .s06_tfidf_rank import run as step9
    from .s07_word_vector_model import run as step10
    from .pipeline_utils import detect_delimiter
except ImportError:
    from s01_1_preprocessing import run as step1
    from s01_2_vocabular import run as step2
    from s01_3_statistics import run as step3
    from s01_4_pos_tag import run as step4
    from s02_preprocessing_gensim import run as step5
    from s03_dtm_tfidf import run as step6
    from s04_cosine import run as step7
    from s05_dtm_tfidf_cos_intervals import run as step8
    from s06_tfidf_rank import run as step9
    from s07_word_vector_model import run as step10
    from pipeline_utils import detect_delimiter


# ---------------------------------------------------------
# Delimiter-Normalisierung
# ---------------------------------------------------------

def normalize_delimiter(delimiter_cfg: str, reference_file: Optional[Path] = None) -> str:
    """
    Normalisiert den Delimiter-Wert aus der Config.
    
    Args:
        delimiter_cfg: Wert aus der Config ("auto", ";", "\\t", etc.)
        reference_file: Optional - Datei für automatische Erkennung
    
    Returns:
        Normalisierter Delimiter
    """
    if delimiter_cfg == "auto":
        if reference_file and reference_file.exists():
            return detect_delimiter(reference_file)
        return ";"  # Fallback
    
    # Escape-Sequenzen verarbeiten
    if delimiter_cfg == "\\t":
        return "\t"
    
    return delimiter_cfg


# ---------------------------------------------------------
# Config-Datei laden
# ---------------------------------------------------------

def load_config(path: Path) -> dict:
    """Lädt die TOML-Konfigurationsdatei."""
    with path.open("rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------
# Pipeline zur Verarbeitung des Korpus
# ---------------------------------------------------------

def run_pipeline_with_cfg(cfg: dict) -> None:
    """
    Führt die Pipeline mit der gegebenen Konfiguration aus.
    
    Wichtig:
    - Delimiter wird automatisch erkannt wenn "auto"
    - Erkannter Delimiter wird an alle Schritte weitergegeben
    """
    
    print("=" * 80)
    print("NLP-PIPELINE v3 - MIT AUTOMATISCHER ERKENNUNG")
    print("=" * 80)

    steps_to_run = cfg.get("run", {}).get("steps", [str(i) for i in range(1, 11)])
    
    # Globaler erkannter Delimiter (wird vom ersten Schritt gesetzt)
    detected_delimiter = None

    # =========================================================================
    # STEP 1: PREPROCESSING
    # =========================================================================
    if "1" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 1: PREPROCESSING")
        print("=" * 80)
        c = cfg["step1_s01_1_preprocessing"]
        
        input_path = Path(c["input_path"])
        delimiter_cfg = c.get("delimiter", "auto")
        
        # Delimiter normalisieren (auto → erkennen)
        delimiter = normalize_delimiter(delimiter_cfg, input_path)
        
        result_delimiter = step1(
            input_path=input_path,
            output_dir=Path(c["output_dir"]),
            delimiter=delimiter,
            replacements_path=Path(c["replacements_path"]),
            stopwords_path=Path(c["stopwords_path"]),
            salat_path=Path(c["salat_path"]),
            hanta_model=c["hanta_model"],
        )
        
        # Speichere den erkannten/verwendeten Delimiter für nachfolgende Schritte
        detected_delimiter = result_delimiter if result_delimiter else delimiter
        
        print(f"✅ Vorverarbeitung abgeschlossen! (Delimiter: {repr(detected_delimiter)})\n")

    # =========================================================================
    # STEP 2: VOKABULAR
    # =========================================================================
    if "2" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 2: VOKABULAR")
        print("=" * 80)
        c = cfg["step2_s01_2_vocabular"]
        
        input_dir = Path(c["input_dir"])
        delimiter_cfg = c.get("delimiter", "auto")
        
        # Wenn bereits erkannt, verwenden; sonst neu erkennen
        if detected_delimiter and delimiter_cfg == "auto":
            delimiter = detected_delimiter
        else:
            # Referenzdatei für Erkennung
            ref_file = input_dir / "korpus_min.csv"
            delimiter = normalize_delimiter(delimiter_cfg, ref_file)
        
        step2(
            input_dir=input_dir,
            output_dir=Path(c["output_dir"]),
            delimiter=delimiter,
            interval_size=c.get("interval_size", 15),
        )
        print("✅ Vokabular ausgelesen!\n")

    # =========================================================================
    # STEP 3: STATISTIK
    # =========================================================================
    if "3" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 3: STATISTIK")
        print("=" * 80)
        c = cfg["step3_s01_3_statistics"]
        
        preprocessed_dir = Path(c["preprocessed_dir"])
        delimiter_cfg = c.get("delimiter", "auto")
        
        if detected_delimiter and delimiter_cfg == "auto":
            delimiter = detected_delimiter
        else:
            ref_file = preprocessed_dir / "korpus_min.csv"
            delimiter = normalize_delimiter(delimiter_cfg, ref_file)
        
        step3(
            preprocessed_dir=preprocessed_dir,
            output_dir=Path(c["output_dir"]),
            delimiter=delimiter,
        )
        print("✅ Statistik ausgelesen!\n")

    # =========================================================================
    # STEP 4: POS-TAGGING
    # =========================================================================
    if "4" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 4: POS-TAGGING")
        print("=" * 80)
        c = cfg["step4_s01_4_pos_tag"]
        step4(
            input_json=Path(c["input_json"]),
            output_csv=Path(c["output_csv"]),
            model=c["model"],
            limit=c["limit"],
        )
        print("✅ POS-Tagging abgeschlossen!\n")

    # =========================================================================
    # STEP 5: PREPROCESSING GENSIM
    # =========================================================================
    if "5" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 5: PREPROCESSING GENSIM (mit optionalen Intervallen)")
        print("=" * 80)
        c = cfg["step5_s02_preprocessing_gensim"]
        
        input_path = Path(c["input_path"])
        delimiter_cfg = c.get("delimiter", "auto")
        
        if detected_delimiter and delimiter_cfg == "auto":
            delimiter = detected_delimiter
        else:
            delimiter = normalize_delimiter(delimiter_cfg, input_path)
        
        intervals = c.get("intervals", None)
        keep_sentence_punct = not c.get("remove_sentence_punct", False)
        
        step5(
            input_path=input_path,
            output_path=Path(c["output_path"]),
            delimiter=delimiter,
            replacements_path=Path(c["replacements_path"]),
            stopwords_path=Path(c["stopwords_path"]),
            salat_path=Path(c["salat_path"]),
            hanta_model=c["hanta_model"],
            keep_sentence_punct=keep_sentence_punct,
            intervals=intervals,
        )
        print("✅ Vorverarbeitung für gensim abgeschlossen!\n")

    # =========================================================================
    # STEP 6: DTM & TF-IDF MATRIZEN
    # =========================================================================
    if "6" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 6: DTM & TF-IDF MATRIZEN")
        print("=" * 80)
        c = cfg["step6_s03_dtm_tfidf"]
        
        input_path = Path(c["input_path"])
        sep_cfg = c.get("sep", "auto")
        
        if detected_delimiter and sep_cfg == "auto":
            sep = detected_delimiter
        else:
            sep = normalize_delimiter(sep_cfg, input_path)
        
        step6(
            input_path=input_path,
            output_dir=Path(c["output_dir"]),
            sep=sep,
        )
        print("✅ DTM und tfidf-Matrizen erstellt!\n")

    # =========================================================================
    # STEP 7: KOSINUS-MATRIZEN
    # =========================================================================
    if "7" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 7: KOSINUS-MATRIZEN")
        print("=" * 80)
        c = cfg["step7_s04_cosine"]
        step7(
            input_path=Path(c["input_path"]),
            output_path=Path(c["output_path"]),
        )
        print("✅ Kosinus-Matrizen erstellt!\n")

    # =========================================================================
    # STEP 8: INTERVALL-MATRIZEN
    # =========================================================================
    if "8" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 8: INTERVALL-MATRIZEN (DTM, TF-IDF, Kosinus)")
        print("=" * 80)
        c = cfg["step8_s05_dtm_tfidf_cos_intervals"]
        
        input_path = Path(c["input_path"])
        sep_cfg = c.get("sep", "auto")
        
        if detected_delimiter and sep_cfg == "auto":
            sep = detected_delimiter
        else:
            sep = normalize_delimiter(sep_cfg, input_path)
        
        step8(
            input_path=input_path,
            dtm_output=Path(c["dtm_output"]),
            cos_output=Path(c["cos_output"]),
            sep=sep,
            interval_size=c.get("interval_size", 15),
        )
        print("✅ Matrizen für Intervalle erstellt!\n")

    # =========================================================================
    # STEP 9: TF-IDF RANGLISTEN
    # =========================================================================
    if "9" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 9: TF-IDF RANGLISTEN")
        print("=" * 80)
        c = cfg["step9_s06_tfidf_rank"]
        step9(
            input_dir=Path(c["input_dir"]),
            output_dir=Path(c["output_dir"]),
            pattern=c.get("pattern", "tfidf"),
            top_n=c["top_n"],
        )
        print("✅ tfidf-Ranglisten erstellt!\n")

    # =========================================================================
    # STEP 10: WORT-VEKTOR-MODELLE
    # =========================================================================
    if "10" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 10: WORT-VEKTOR-MODELLE (adaptive Parameter)")
        print("=" * 80)
        c = cfg["step10_s07_word_vector_model"]
        
        delimiter_cfg = c.get("delimiter", "auto")
        
        if detected_delimiter and delimiter_cfg == "auto":
            delimiter = detected_delimiter
        else:
            delimiter = normalize_delimiter(delimiter_cfg)
        
        step10(
            input_dir=Path(c["input_dir"]),
            output_dir=Path(c["output_dir"]),
            pattern=c["pattern"],
            delimiter=delimiter,
        )
        print("✅ Wort-Vektor-Modelle erstellt!\n")

    print("\n" + "=" * 80)
    print("✅ PIPELINE ABGESCHLOSSEN!")
    print("=" * 80)


# ---------------------------------------------------------
# Main-Funktion
# ---------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NLP-Pipeline FaDe:Live v3 - Mit automatischer Erkennung",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features v3:
  • Automatische Delimiter-Erkennung (Fallback: ";")
  • Automatische Metadaten-Handhabung
  • Dynamische Intervall-Generierung
  • Flexible ID-/Jahr-Spalten-Erkennung

Beispiele:
  # Alle Schritte ausführen
  python -m fadelive.pipeline --config config/fadelive_v3.toml
  
  # Nur bestimmte Schritte
  python -m fadelive.pipeline --config config/fadelive_v3.toml --steps 1 2 3
  
  # Nur Gensim-Preprocessing und Word2Vec
  python -m fadelive.pipeline --steps 5 10
        """
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/fadelive_v3.toml"),
        help="Pfad zur TOML-Konfigurationsdatei",
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        help="Optional: Nur bestimmte Schritte ausführen, z.B. 1 2 3 für Preprocessing+Vocab+Stats",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.steps is not None:
        cfg.setdefault("run", {})["steps"] = args.steps

    run_pipeline_with_cfg(cfg)


if __name__ == "__main__":
    main()
