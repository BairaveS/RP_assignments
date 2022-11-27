import transformers
import streamlit as st
import time

@st.cache(allow_output_mutation=True)
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

def en_to_es(input_text):
    input_text = input_text.strip()
    if len(input_text) > 0:
        transation = en_es_translator(input_text)[0]["generated_text"]
        return transation.strip()
    else:
        return ""

def es_to_en(input_text):
    input_text = input_text.strip()
    if len(input_text) > 0:
        transation = es_en_translator(input_text)[0]["generated_text"]
        return transation.strip()
    else:
        return ""

st.title("English-Spanish Translation App")

direction = st.selectbox("Direction", ["English -> Spanish", "Spanish -> English"])

input_text = st.text_area("Input text", value="")

start_time = time.time()
if direction == "English -> Spanish":
    translation = en_to_es(input_text)
else:
    translation = es_to_en(input_text)
end_time = time.time()

time_taken = str(round(end_time-start_time,2))
    
st.markdown("**Translation**: "+translation)

st.markdown("Time taken: "+str(time_taken)+"s")

