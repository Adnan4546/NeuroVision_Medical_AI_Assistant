import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import os

# --- STABLE LANGCHAIN IMPORTS ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. API SETUP
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#os.environ["GROQ_API_KEY"] =
llm = ChatGroq(model="llama-3.1-8b-instant")

st.set_page_config(page_title="NeuroVision Medical AI", layout="wide", page_icon="🧠")

# 2. INITIALIZE SESSION STATE (Fixes the AttributeError)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "tumor_type" not in st.session_state:
    st.session_state.tumor_type = None

CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']


# ==========================================
# 3. CACHE MODELS & HELPERS
# ==========================================
@st.cache_resource
def load_classification_model():
    return tf.keras.models.load_model('my_model.keras')


classifier_model = load_classification_model()


def generate_gradcam(img_array, model, pred_index):
    vgg_base = model.get_layer("vgg16")
    last_conv_layer_name = "block5_conv3"
    vgg_grad_model = tf.keras.Model(
        inputs=vgg_base.input,
        outputs=[vgg_base.get_layer(last_conv_layer_name).output, vgg_base.output]
    )
    classifier_input = tf.keras.Input(shape=vgg_base.output.shape[1:])
    x = classifier_input
    for layer in model.layers[1:]:
        x = layer(x)
    classifier_model_head = tf.keras.Model(inputs=classifier_input, outputs=x)

    with tf.GradientTape() as tape:
        conv_outputs, vgg_output = vgg_grad_model(img_array)
        tape.watch(conv_outputs)
        predictions = classifier_model_head(vgg_output)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.title("🏥 NeuroVision Medical AI Platform")

tab1, tab2 = st.tabs(["🧠 MRI Diagnosis & XAI", "📄 LangChain Document Analyzer"])

# ------------------------------------------
# TAB 1: MRI Diagnosis & Unified Chat
# ------------------------------------------
with tab1:
    st.markdown("### 🧠 CortexInsight MRI Analysis")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="mri_uploader")

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        img_resized = img.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing neural patterns..."):
            preds = classifier_model.predict(img_array)
            pred_index = np.argmax(preds[0])
            pred_class = CLASS_LABELS[pred_index]
            confidence = np.max(preds)

            # Store results in session state
            st.session_state.tumor_type = pred_class
            heatmap = generate_gradcam(img_array, classifier_model, pred_index)

            # Heatmap processing
            original_img_array = tf.keras.utils.img_to_array(img_resized)
            heatmap_uint8 = np.uint8(255 * heatmap)
            jet = mpl.colormaps["jet"]
            jet_colors = jet(np.arange(256))[:, :3]
            jet_heatmap = jet_colors[heatmap_uint8]
            jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap).resize((224, 224))
            jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
            superimposed_img = tf.keras.utils.array_to_img(jet_heatmap * 0.4 + original_img_array)

        st.success(f"**Analysis Complete:** {pred_class.upper()} detected ({confidence * 100:.2f}% confidence)")

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_resized, caption="Original MRI Scan", use_container_width=True)
        with col2:
            st.image(superimposed_img, caption="Grad-CAM XAI Visualization", use_container_width=True)

        st.divider()

        # --- UNIFIED CHAT LOGIC ---
        if st.session_state.tumor_type:
            st.subheader(f"💬 CortexInsight Consultation: {st.session_state.tumor_type.title()}")

            # 1. Display Chat History
            for msg in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(msg["patient"])
                with st.chat_message("assistant"):
                    st.write(msg["doctor"])

            # 2. Unified Chat Input (Independent of the loop)
            if user_query := st.chat_input("Ask about the diagnosis, tumor location, or the XAI heatmap..."):
                with st.chat_message("user"):
                    st.write(user_query)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing clinical data and XAI focus..."):
                        unified_prompt = f"""
                        You are a highly skilled Neuro-Oncology Specialist and Radiologist at CortexInsight. 
                        The AI system has analyzed an MRI scan and found:
                        - PREDICTED TUMOR TYPE: {st.session_state.tumor_type}
                        - EXPLAINABLE AI (XAI): A Grad-CAM heatmap showing focus areas.

                        YOUR TASK:
                        1. If the user asks about the heatmap or 'where' the tumor is: Explain that red/yellow regions indicate model focus and correlate this with the anatomy of a {st.session_state.tumor_type}.
                        2. If the user asks general medical questions: Provide info on symptoms/treatments for {st.session_state.tumor_type}.
                        3. Always remain professional and empathetic.

                        PATIENT QUESTION: "{user_query}"
                        """

                        try:
                            response = llm.invoke(unified_prompt)
                            answer = response.content
                            st.write(answer)
                            st.session_state.chat_history.append({"patient": user_query, "doctor": answer})
                            st.rerun()  # Refresh to show new messages immediately
                        except Exception as e:
                            st.error(f"Consultation Error: {e}")
# TAB 2: PDF RAG
# ------------------------------------------
with tab2:
    st.markdown("### 📄 Professional Medical Document Analysis")
    uploaded_pdf = st.file_uploader("Upload a Medical PDF", type="pdf")

    if uploaded_pdf:
        temp_pdf_path = "temp_uploaded.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        with st.spinner("Indexing Document..."):
            loader = PyPDFLoader(temp_pdf_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            final_documents = text_splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
            vector_db = FAISS.from_documents(final_documents, embeddings)

        st.success("Ready for questions.")
        pdf_query = st.text_input("What would you like to know from the document?")

        if pdf_query:
            search_results = vector_db.similarity_search(pdf_query, k=3)
            context_text = "\n\n".join([doc.page_content for doc in search_results])
            full_prompt = f"Context: {context_text}\n\nQuestion: {pdf_query}"
            response = llm.invoke(full_prompt)
            st.info(response.content)