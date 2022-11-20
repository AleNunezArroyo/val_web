import tensorflow as tf
import torch
from transformers import BertForSequenceClassification,BertTokenizerFast
import streamlit as st
import torch.nn.functional as F
import unidecode
import nltk
nltk.download('stopwords')
import gensim
import pandas as pd
from nltk.corpus import stopwords
from PIL import Image
from pathlib import Path
import os
current_directory = Path(__file__).parent #Get current directory


@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = BertTokenizerFast.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
    model = BertForSequenceClassification.from_pretrained("AleNunezArroyo/BETO_BolivianFN")
    return tokenizer,model

@st.cache(allow_output_mutation=True)
def get_extra():
    nltk.download("stopwords")
    stop_words = stopwords.words('spanish')
    image = open(os.path.join(current_directory, 'file/image.jpg'), 'rb')
    image = Image.open(image)
    return stop_words, image


tokenizer,model = get_model()
stop_words, image = get_extra()

st.image(image, caption='Estudiante: Alejandro Núñez Arroyo | Tutor: Ing. Guillermo Sahonero')

# st.title("Ejemplos de texto:")
# option = st.selectbox(
#     'Ejemplos de datos de prueba:',
#     ('Dirigentes del transporte cruceño alentaron a salir con palos a desbloquear | Falso', 
#      'UTO producirá dióxido de cloro y su Rector denuncia que los insumos ya subieron | Verdadero'))

user_input = st.text_area('Ingresar texto para revisión', value="Escribe aquí el extracto a verificar o prueba un ejemplo...", key=1)
pre_pro = st.radio(
    "Seleccione el filtrado:",
    ('Con preprocesamiento', 'Sin preprocesamiento'))
button = st.button("Analizar")

st.sidebar.title("Información adicional")
st.sidebar.info('Descarga la guía de usuario:')
link='[ENLACE](https://github.com/AlejandroNunezArroyo/test_web_/blob/main/file/Protocolo.pdf)'
st.sidebar.markdown(link,unsafe_allow_html=True)
st.sidebar.header('Ejemplos falso: ')
st.sidebar.markdown('Vacunas genéticas provocan daño al ser humano')
st.sidebar.markdown('Red Uno publicó que Iván Arias y Eva Copa hayan realizado una alianza')

st.sidebar.header('Ejemplos verdaderos: ')
st.sidebar.markdown('Tarija Cuatro menores fugaron del centro Oasis')
st.sidebar.markdown('Alanez exclama que tiene los pantalones bien amarrados y sus cocaleros abren marcha pacífica')

st.sidebar.header('No hace inferencia: ')
st.sidebar.markdown('Dirigentes del transporte cruceño alentaron a salir con palos a desbloquear')

def joined_data(text):
    try:
        text = " ".join(text)
        return (text)
    except:
        pass
    
def data_filter(text):
    result = []
    # Convierte en una lista de tokens en minúscula
    # https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess
    try:
        for token in gensim.utils.simple_preprocess(text):
            # En caso de que el token no esté en stop_words
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2 and token not in stop_words:
                # remove ascents
                token = unidecode.unidecode(token)
                # transform to root word
                # token = stemmer.stem(token)
                result.append(token)
        return (joined_data(result))
    except:
        print(text)
        pass

MAX_SEQ_LEN = 21 

if user_input and button:
    if (pre_pro == 'Con preprocesamiento'):
        user_input = data_filter(user_input)
    else:
        user_input = user_input
    
    st.write("Texto de entrada al sistema: ",user_input, key="3")
    st.write("Longitud: ",len(user_input.split(' ')), key="4")
    encoded_review = tokenizer.encode_plus(
        user_input,
        max_length=MAX_SEQ_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded_review['input_ids']
    attention_mask = encoded_review['attention_mask']
    model_out = model(input_ids,attention_mask)
    all_logits = torch.nn.functional.log_softmax(model_out.logits, dim=1)
    probs = F.softmax(all_logits, dim=1)
    p_ = torch.squeeze(probs)
    p_ = p_.tolist()
    col1_, col2_ = st.columns(2)
    if (round(p_[0], 5) > 0.9057):
        with col1_:
            st.success('Probalidad de Verdadero: ', icon="✅")
        with col2_:
            st.header((round(p_[0], 3)*100), ' %')
        st.info('Predicción realizada con éxito')
        
    elif (round(p_[1], 5) > 0.8809):
        with col1_:
            st.error('El texto de entrada necesita atención. Probalidad de falso: ', icon="🚨")
        with col2_:
            st.header((round(p_[1], 3)*100), ' %')
        st.info('Predicción realizada con éxito')
    else:
        st.info('Intenta con otra noticia')
        st.write(round(p_[0], 5), (round(p_[0], 5)))


