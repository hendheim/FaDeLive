"""mit dem Script werden alle Schritte des NLP des Korpus FaDe:Live ausgeführt 

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
from pathlib import Path

def run_pipeline():

    print("Starte Pipeline...")

# s01_1_preprocessing: Vorverarbeitung

    print("Starte Vorverarbeitung...")
    step1(
    input_path=Path("data/raw/korpus.csv"),
    output_dir=Path("output/processed_corpus"),
    delimiter=";",
    replacements_path=Path("resources/replacements_v1.json"),
    stopwords_path=Path("resources/stopwords_v1.txt"),
    salat_path=Path("resources/ocr_post-correction_dictionary.txt"),
    hanta_model="morphmodel_ger.pgz",)
    print("Vorverarbeitung abgeschlossen!")

# s01_2_vocabular: Erzeugung des Vokabulars

    print("Lese das Vokabular aus...")
    step2(
    input_dir=Path("output/processed_corpus"),
    output_dir=Path("output/vocabular"),
    delimiter=";",)
    print("Vokabular ausgelesen!")

# s01_3_statistics: Erzeugung von Statistiken

    print("Lese die Statistik aus...")
    step3(
    preprocessed_dir=Path("output/processed_corpus"),
    output_dir=Path("output/statistics"),
    delimiter=";",)
    print("Statistik ausgelesen!")

# s01_4_pos-tag: POS-Tagging der Top-5000 Ausdrücke des maximal vorverarbeiteten Korpus 

    print("Starte POS-Tagging...")
    step4(
    input_json=Path("output/vocabular/vocab_full_stop.json"),
    output_csv=Path("output/vocabular/vocab_top5000_stop_pos.csv"),
    model="de_core_news_lg",
    limit=5000,)
    print("POS-Tagging abgeschlossen!")

# s02_preprocessing_gensim: Erzeugung der Vorverarbeitungsstufe TXT (gen) für 's07_word_vector_model'

    print("Starte Vorverarbeitung des Korpus für ein Wort-Vektor-Modell...")
    step5(
    input_path=Path("output/processed_corpus/korpus_min.csv"),
    output_path=Path("output/processed_corpus/korpus_gen.csv"),
    delimiter=";",
    replacements_path=Path("resources/replacements_v1.json"),
    stopwords_path=Path("resources/stopwords_v1.txt"),
    salat_path=Path("resources/ocr_post-correction_dictionary.txt"),
    hanta_model="morphmodel_ger.pgz",)
    print("Matrizen erzeugt!")

# s03_dtm_tfidf: Erzeugung der dtm- und tfidf-Matrizen mit Metadaten

    print("Erzeuge Dokument-Term-Matrizen und tfidf-Matrizen...")
    step6(
    input_path=Path("output/processed_corpus/korpus_stop.csv"),
    output_dir=Path("output/dtm_tfidf_stop"),
    sep= ";",
    )
    print("Matrizen erzeugt!")

# s04_cosine: Erzeugung der Kosinus-Matrizen

    print("Erzeuge Kosinus-Matrizen...")
    step7(
    input_path=Path("output/dtm_tfidf_stop/tfidf-2000.csv"),
    output_path=Path("output/cosine/cosine_tfidf2000.csv"),
    )
    print("Kosinus-Matrizen erzeugt!")


# s05_dtm_tfidf_cos_intervals: s03 und s04 für Intervalle

    print("Erzeuge Matrizen für Intervalle...")
    step8(
    input_path=Path("output/processed_corpus/korpus_stop.csv"),
    dtm_output=Path("output/intervals/dtm_tfidf_stop"),
    cos_output=Path("output/intervals/cosine_stop"),
    sep=";"
    )
    print("Matrizen für Intervalle erzeugt!")


# s06_tfidf_rank: Erzeugung der tfidf-Ranglisten des Vokabulars und der Texte

    print("Lese die tfidf-Ranglisten für das Vokabular und für die Dokumente aus...")
    step9(
    input_dir = Path("output"),
    output_dir = Path("output/tfidf_rank"),
    top_n = 2000,
    )
    print("tfidf-Ranglisten ausgelesen!")


# s07_word_vector_model: Erzeugung des Wort-Vektor-Modells des Korpus mit gensim 

    print("Erzeuge ein Wort-Vektor-Modell des Korpus...")
    step10(
    input_dir = Path("output/processed_corpus"),
    output_dir = Path("output/word2vec_models"),
    pattern = "korpus_gen*.csv",
    )
    print("Wort-Vektor-Modell erzeugt")