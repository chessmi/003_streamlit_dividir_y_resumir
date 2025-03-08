import streamlit as st
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import pandas as pd
from io import StringIO

# Función para cargar el modelo de lenguaje (LLM)
def cargar_LLM(api_key_openai):
    # Make sure your openai_api_key is set as an environment variable
    llm = ChatOpenAI(temperature=0, openai_api_key=api_key_openai)
    return llm


# Configuración de la página
st.set_page_config(page_title="Resumidor de Texto con IA")
st.header("Resumidor de Texto con Inteligencia Artificial")


#Intro: instructions
col1, = st.columns(1)

with col1:
    st.markdown("ChatGPT no puede resumir textos largos. Ahora puedes hacerlo con esta aplicación.")


# Entrada de clave API de OpenAI
st.markdown("## Ingresa tu clave de API de OpenAI")

def get_openai_api_key():
    input_text = st.text_input(label="Clave API de OpenAI", placeholder="Ejemplo: sk-2twmA8tfCb8un4...", key="api_key_openai_input", type="password")
    return input_text

openai_api_key = get_openai_api_key()


# Carga de archivos
st.markdown("## Sube un archivo de texto para resumir")

archivo_subido = st.file_uploader("Elige un archivo", type="txt")

       
# Output
st.markdown("### Ver resumen completo:")

if archivo_subido is not None:
    # To read file as bytes:
    bytes_data = archivo_subido.getvalue()
    #st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(archivo_subido.getvalue().decode("utf-8"))
    #st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    #st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    #dataframe = pd.read_csv(archivo_subido)
    #st.write(dataframe)

    texto_archivo = string_data

    if len(texto_archivo.split(" ")) > 20000:
        st.write("El archivo es demasiado largo. El límite es de 20,000 palabras.")
        st.stop()

    if texto_archivo:
        if not openai_api_key:
            st.warning('Por favor, ingresa tu clave API de OpenAI. \
            Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', 
            icon="⚠️")
            st.stop()

    divisor_texto = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], 
        chunk_size=5000, 
        chunk_overlap=350
        )

    documentos_divididos = divisor_texto.create_documents([texto_archivo])

    llm = cargar_LLM(api_key_openai=openai_api_key)

    # Crear prompt para resumir en español
    resumen_prompt = PromptTemplate(
        input_variables=["texto"],
        template="Resume el siguiente texto en español:\n\n{texto}"
    )
    
    cadena_resumen  = load_summarize_chain(
        llm=llm, 
        chain_type="map_reduce",
        map_prompt=resumen_prompt,  # Usa el prompt para cada fragmento
        combine_prompt=resumen_prompt  # Usa el mismo prompt para combinar los resúmenes
    )

    resumen_generado = cadena_resumen.run(documentos_divididos)

    st.write(resumen_generado)
