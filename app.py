import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from groq import Groq

# Configuración de la página
st.set_page_config(page_title="Agrodetect - Detección Foliar - Ángel Gómez", layout="wide")

# Título y descripción
st.markdown("### Captura de Imagen Foliar")
st.caption("Posicione la hoja de café bajo luz natural. El sistema detectará automáticamente signos de Roya, Cercospora o Plagas.")

# Dividir la interfaz en dos columnas
col_izq, col_der = st.columns([1.2, 1.8], gap="large")

# Cargar modelo una sola vez
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model('/content/dataset_cafe/modelo_enfermedades_cafe.keras')

model = cargar_modelo()

with col_izq:
    st.markdown("**Subir archivo** &nbsp;&nbsp;&nbsp;&nbsp; **Usar cámara**")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Imagen cargada', use_column_width=True)

with col_der:
    if uploaded_file is not None:
        # Preprocesamiento
        img = image.load_img(uploaded_file, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0
        
        # Predicción
        predictions = model.predict(x)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        clases = ['leaf_miner', 'otros', 'phoma', 'rust']
        nombres_clases = {'leaf_miner': 'Minador de la Hoja', 'phoma': 'Phoma', 'rust': 'Roya', 'otros': 'Hoja Sana'}
        resultado = clases[predicted_class]
        
        # Visualización de resultados
        st.markdown(f"### Diagnóstico: {nombres_clases.get(resultado, resultado)}")
        st.metric("Nivel de confianza", f"{confidence:.2f}%")
        
        st.markdown("---")
        st.markdown("**ORIENTACIÓN Y MANEJO PREVENTIVO (IA)**")
        
        # Integración con Groq usando tu API key
        try:
            client = Groq(api_key="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")
            prompt = f"Actúa como un ingeniero agrónomo experto en café. Analiza el diagnóstico: {resultado}. Dame recomendaciones técnicas específicas para el manejo preventivo y control de esta enfermedad en el cultivo de café."
            
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
            )
            st.write(chat_completion.choices[0].message.content)
        except Exception as e:
            st.error(f"Error al conectar con la API de Groq: {e}")
            
    else:
        st.info("Sube una imagen para realizar el diagnóstico y obtener recomendaciones técnicas.")

st.markdown("---")
st.caption("(c) 2026 AGRODETECT • SOPORTE IHCAFE")