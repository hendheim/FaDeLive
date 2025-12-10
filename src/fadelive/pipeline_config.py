"""mit dem Script werden alle Schritte des NLP entsprechend der config-Datei ausgeführt

NLP-Pipeline v3 - Finale Version mit flexibler Metadaten-Handhabung:
    - s01_1_preprocessing: Vorverarbeitung von data/raw/korpus.csv und Erzeugung von gesäuberten und normalisierten Textvarianten (TXT [min]), lemmatisierten Textvarianten (TXT [lem]) und von Stoppwörter gereinigten Textvarianten (TXT [stop])
    - s01_2_vocabular: Erzeugung des Vokabulars (mit automatischer Metadaten-Erkennung)
    - s01_3_statistics: Erzeugung von Statistiken (nur für vorhandene Metadaten-Spalten)
    - s01_4_pos_tag: POS-Tagging der Top-5000 Ausdrücke des maximal vorverarbeiteten Korpus
    - s02_preprocessing_gensim: Erzeugung der Vorverarbeitungsstufe TXT (gen) für 's07_word_vector_model' (mit optionaler Intervall-Unterstützung)
    - s03_dtm_tfidf: Erzeugung der dtm- und tfidf-Matrizen mit Metadaten (automatische Erkennung)
    - s04_cosine: Erzeugung der Kosinus-Matrizen (intelligente Feature-Trennung)
    - s05_dtm_tfidf_cos_intervals: s03 und s04 für Zeitintervalle (year_first hat Vorrang)
    - s06_tfidf_rank: Erzeugung der tfidf-Ranglisten des Vokabulars und der Texte (automatische Metadaten-Erkennung)
    - s07_word_vector_model: Erzeugung des Wort-Vektor-Modells des Korpus mit gensim (adaptive Parameter, automatische Intervall-Erkennung)

Wichtige Änderungen v3:
    - Alle Module arbeiten mit flexiblen Metadaten (alles außer 'content')
    - year/year_first werden speziell behandelt (year_first hat Vorrang)
    - Step 5: Optionale Zeitintervalle für separate Gensim-Dateien
    - Step 10: Automatische Parameter-Anpassung an Korpusgröße
    - Step 10: Automatische Erkennung und Verarbeitung von Intervall-Dateien
"""

from pathlib import Path
import tomllib  
import argparse

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


# ---------------------------------------------------------
# config-Datei laden
# ---------------------------------------------------------

def load_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------
# Pipeline zur Verarbeitung des Korpus 
# ---------------------------------------------------------

def run_pipeline_with_cfg(cfg: dict) -> None:
    
    print("=" * 80)
    print("NLP-PIPELINE v3 - FADELIVE")
    print("=" * 80)

    steps_to_run = cfg.get("run", {}).get("steps", [str(i) for i in range(1, 11)])

    if "1" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 1: PREPROCESSING")
        print("=" * 80)
        c = cfg["step1_s01_1_preprocessing"]
        step1(
            input_path=Path(c["input_path"]),
            output_dir=Path(c["output_dir"]),
            delimiter=c["delimiter"],
            replacements_path=Path(c["replacements_path"]),
            stopwords_path=Path(c["stopwords_path"]),
            salat_path=Path(c["salat_path"]),
            hanta_model=c["hanta_model"],
        )
        print("✅ Vorverarbeitung abgeschlossen!\n")

    if "2" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 2: VOKABULAR")
        print("=" * 80)
        c = cfg["step2_s01_2_vocabular"]
        step2(
            input_dir=Path(c["input_dir"]),
            output_dir=Path(c["output_dir"]),
            delimiter=c["delimiter"],
        )
        print("✅ Vokabular ausgelesen!\n")

    if "3" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 3: STATISTIK")
        print("=" * 80)
        c = cfg["step3_s01_3_statistics"]
        step3(
            preprocessed_dir=Path(c["preprocessed_dir"]),
            output_dir=Path(c["output_dir"]),
            delimiter=c["delimiter"],
        )
        print("✅ Statistik ausgelesen!\n")

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

    if "5" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 5: PREPROCESSING GENSIM (mit optionalen Intervallen)")
        print("=" * 80)
        c = cfg["step5_s02_preprocessing_gensim"]
        
        # Intervalle aus Config lesen (optional)
        intervals = c.get("intervals", None)
        
        # keep_sentence_punct aus Config lesen (Standard: True, außer remove_sentence_punct gesetzt)
        keep_sentence_punct = not c.get("remove_sentence_punct", False)
        
        step5(
            input_path=Path(c["input_path"]),
            output_path=Path(c["output_path"]),
            delimiter=c["delimiter"],
            replacements_path=Path(c["replacements_path"]),
            stopwords_path=Path(c["stopwords_path"]),
            salat_path=Path(c["salat_path"]),
            hanta_model=c["hanta_model"],
            keep_sentence_punct=keep_sentence_punct,
            intervals=intervals,
        )
        print("✅ Vorverarbeitung für gensim abgeschlossen!\n")
    
    if "6" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 6: DTM & TF-IDF MATRIZEN")
        print("=" * 80)
        c = cfg["step6_s03_dtm_tfidf"]
        step6(
            input_path=Path(c["input_path"]),
            output_dir=Path(c["output_dir"]),
            sep=c["sep"],
        )
        print("✅ DTM und tfidf-Matrizen erstellt!\n")

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

    if "8" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 8: INTERVALL-MATRIZEN (DTM, TF-IDF, Kosinus)")
        print("=" * 80)
        c = cfg["step8_s05_dtm_tfidf_cos_intervals"]
        step8(
            input_path=Path(c["input_path"]),
            dtm_output=Path(c["dtm_output"]),
            cos_output=Path(c["cos_output"]),
            sep=c["sep"],
        )
        print("✅ Matrizen für Intervalle erstellt!\n")

    if "9" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 9: TF-IDF RANGLISTEN")
        print("=" * 80)
        c = cfg["step9_s06_tfidf_rank"]
        step9(
            input_dir=Path(c["input_dir"]),
            output_dir=Path(c["output_dir"]),
            top_n=c["top_n"],
        )
        print("✅ tfidf-Ranglisten erstellt!\n")

    if "10" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 10: WORT-VEKTOR-MODELLE (adaptive Parameter)")
        print("=" * 80)
        c = cfg["step10_s07_word_vector_model"]
        step10(
            input_dir=Path(c["input_dir"]),
            output_dir=Path(c["output_dir"]),
            pattern=c["pattern"],
            delimiter=c.get("delimiter", ";"),
        )
        print("✅ Wort-Vektor-Modelle erstellt!\n")

    print("\n" + "=" * 80)
    print("✅ PIPELINE ABGESCHLOSSEN!")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NLP-Pipeline FaDe:Live v3 - Finale Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Wichtige Features v3:
  • Flexible Metadaten-Handhabung (alle Spalten außer 'content')
  • year_first hat Vorrang vor year bei Zeitanalysen
  • Step 5: Optionale Zeitintervalle für Gensim-Preprocessing
  • Step 10: Automatische Parameter-Anpassung an Korpusgröße
  • Step 10: Automatische Intervall-Erkennung

Beispiele:
  # Alle Schritte ausführen
  python -m fadelive.pipeline --config config/fadelive_v4.toml
  
  # Nur bestimmte Schritte
  python -m fadelive.pipeline --config config/fadelive_v4.toml --steps 5 10
  
  # Nur Gensim-Preprocessing und Word2Vec
  python -m fadelive.pipeline --steps 5 10
        """
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/fadelive_v4.toml"),
        help="Pfad zur TOML-Konfigurationsdatei",
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        help="Optional: Nur bestimmte Schritte ausführen, z.B. 5 10 für Gensim+Word2Vec",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.steps is not None:
        cfg.setdefault("run", {})["steps"] = args.steps

    run_pipeline_with_cfg(cfg)


if __name__ == "__main__":
    main()
