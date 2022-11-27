import transformers

def load_models():
    # English to Spanish
    EN_ES_MODEL = "Helsinki-NLP/opus-mt-en-es"
    en_es_tokenizer = transformers.AutoTokenizer.from_pretrained(EN_ES_MODEL)
    en_es_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(EN_ES_MODEL)
    en_es_translator = transformers.pipeline("text2text-generation", model=en_es_model, tokenizer=en_es_tokenizer)
    # Spanish to English
    ES_EN_MODEL = "Helsinki-NLP/opus-mt-es-en"
    es_en_tokenizer = transformers.AutoTokenizer.from_pretrained(ES_EN_MODEL)
    es_en_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(ES_EN_MODEL)
    es_en_translator = transformers.pipeline("text2text-generation", model=es_en_model, tokenizer=es_en_tokenizer)
    return en_es_translator, es_en_translator
    
en_es_translator, es_en_translator = load_models()

