"""mit dem Script werden alle Schritte des NLP entsprechend der config-Datei ausgeführt

NLP: 
    - s01_1_preprocessing: Vorverarbeitung von data/rwa/korpus.csv und Erzeugung von gesäuberten und normalisierten Textvarianten (TXT [min]), lemmatisierten Textvarianten (TXT [lem]) und von Stoppwörter gereinigten Textvarianten (TXT [stop])
    - s01_2_vocabular: Erzeugung des Vokabulars
    - s01_3_statistics: Erzeugung von Statistiken
    - s01_4_pos_tag: POS-Tagging der Top-5000 Ausdrücke des maximal vorverarbeiteten Korpus 
    - s02_preprocessing_gensim: Erzeugung der Vorverarbeitungsstufe TXT (gen) für 's07_word_vector_model'
    - s03_dtm_tfidf: Erzeugung der dtm- und tfidf-Matrizen mit Metadaten
    - s04_cosine: Erzeugung der Kosinus-Matrizen
    - s05_dtm_tfidf_cos_intervals: s03 und s04 für Intervalle
    - s06_tfidf_rank: Erzeugung der tfidf-Ranglisten des Vokabulars und der Texte
    - s07_word_vector_model: Erzeugung des Wort-Vektor-Modells des Korpus mit gensim 

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
    
    print("Starte Pipeline...")

    steps_to_run = cfg.get("run", {}).get("steps", [str(i) for i in range(1, 11)])

    if "1" in steps_to_run:
        print("Starte Vorverarbeitung...")
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
        print("Vorverarbeitung abgeschlossen!")

    if "2" in steps_to_run:
        print("Lese das Vokabular aus...")
        c = cfg["step2_s01_2_vocabular"]
        step2(
            input_dir=Path(c["input_dir"]),
            output_dir=Path(c["output_dir"]),
            delimiter=c["delimiter"],
        )
        print("Vokabular ausgelesen!")

    if "3" in steps_to_run:
        print("Lese die Statistik aus...")
        c = cfg["step3_s01_3_statistics"]
        step3(
            preprocessed_dir=Path(c["preprocessed_dir"]),
            output_dir=Path(c["output_dir"]),
            delimiter=c["delimiter"],
        )
        print("Statistik ausgelesen!")

    if "4" in steps_to_run:
        print("Starte POS-Tagging...")
        c = cfg["step4_s01_4_pos_tag"]
        step4(
            input_json=Path(c["input_json"]),
            output_csv=Path(c["output_csv"]),
            model=c["model"],
            limit=c["limit"],
        )
        print("POS-Tagging abgeschlossen!")

    if "5" in steps_to_run:
        print("Starte Vorverarbeitung für gensim...")
        c = cfg["step5_s02_preprocessing_gensim"]
        step5(
            input_path=Path(c["input_path"]),
            output_path=Path(c["output_path"]),
            delimiter=c["delimiter"],
            replacements_path=Path(c["replacements_path"]),
            stopwords_path=Path(c["stopwords_path"]),
            salat_path=Path(c["salat_path"]),
            hanta_model=c["hanta_model"]
        )
        print("Vorverarbeitung für gensim abgeschlossen!")
    
    if "6" in steps_to_run:
        print("Erstellung der DTM und tfidf-Matrizen...")
        c = cfg["step6_s03_dtm_tfidf"]
        step6(
            input_path=Path(c["input_path"]),
            output_dir=Path(c["output_dir"]),
            sep=c["sep"],
        )
        print("DTM und tfidf-Matrizen erstell´t!")

    if "7" in steps_to_run:
        print("Erstellung Kosinus-Matrizen...")
        c = cfg["step7_s04_cosine"]
        step7(
            input_path=Path(c["input_path"]),
            output_path=Path(c["output_path"]),
        )
        print("Kosinus-Matrizen erstellt!")


    if "8" in steps_to_run:
        print("Erstellung DTM, tfidf- und Kosinus-Matirzen für Intervalle...")
        c = cfg["step8_s05_dtm_tfidf_cos_intervals"]
        step8(
            input_path=Path(c["input_path"]),
            dtm_output=Path(c["dtm_output"]),
            cos_output=Path(c["cos_output"]),
            sep=c["sep"],
        )
        print("Matrizen für Intervalle erstellt!")

    if "9" in steps_to_run:
        print("Erstellung der tfidf-Ranglisten...")
        c = cfg["step9_s06_tfidf_rank"]
        step9(
            input_dir=Path(c["input_dir"]),
            output_dir=Path(c["output_dir"]),
            top_n=c["top_n"],
        )
        print("tfidf-Ranglisten erstellt!")

    if "10" in steps_to_run:
        print("Erstellung des Wort-Vektor-Modells...")
        c = cfg["step10_s07_word_vector_model"]
        step10(
            input_dir=Path(c["input_dir"]),
            output_dir=Path(c["output_dir"]),
            pattern=c["pattern"],
        )
        print("Wort-Vektor-Modell erstellt!")

def main() -> None:
    parser = argparse.ArgumentParser(description="NLP-Pipeline FaDe:Live")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/fadelive_v3.toml"),
        help="Pfad zur TOML-Konfigurationsdatei",
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        help="Optional: Nur bestimmte Schritte ausführen, z.B. 4 6 für POS+TFIDF",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.steps is not None:
        cfg.setdefault("run", {})["steps"] = args.steps

    run_pipeline_with_cfg(cfg)


if __name__ == "__main__":
    main()