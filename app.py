import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
import time
import pandas as pd
import plotly.express as px
from datetime import datetime
from io import BytesIO
import os

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="wide"
)

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource
def load_my_model():
    return load_model("waste_classification_model.keras")

model = load_my_model()

# ==========================================================
# CLASS LABELS
# ==========================================================
class_names = ['Organic', 'Recyclable']

# ==========================================================
# HISTORY FILE
# ==========================================================
history_file = "prediction_history.csv"

# Create CSV file if not exists
if not os.path.exists(history_file):

    history_df = pd.DataFrame(
        columns=["Prediction", "Confidence (%)", "Time"]
    )

    history_df.to_csv(history_file, index=False)

# ==========================================================
# LOAD HISTORY FROM CSV
# ==========================================================
def load_history():

    if os.path.exists(history_file):

        history_df = pd.read_csv(history_file)

        return history_df.to_dict('records')

    return []

# ==========================================================
# SESSION STATES
# ==========================================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "history" not in st.session_state:
    st.session_state.history = load_history()

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# File uploader reset key
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ==========================================================
# BACKGROUND FUNCTION
# ==========================================================
def set_bg(image_file):

    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image:
            linear-gradient(
                rgba(0,0,0,0.55),
                rgba(0,0,0,0.55)
            ),
            url("data:image/jpg;base64,{encoded}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        .main-title {{
            text-align: center;
            font-size: 60px;
            font-weight: 800;
            color: white;
            text-shadow: 2px 2px 15px rgba(0,0,0,0.7);
        }}

        .sub-title {{
            text-align: center;
            color: #f1f1f1;
            font-size: 22px;
            margin-bottom: 40px;
        }}

        .glass {{
            background: rgba(255,255,255,0.12);
            padding: 35px;
            border-radius: 25px;
            backdrop-filter: blur(15px);
            box-shadow: 0px 8px 32px rgba(0,0,0,0.35);
        }}

        .result-card {{
            background: rgba(255,255,255,0.12);
            padding: 40px;
            border-radius: 30px;
            backdrop-filter: blur(15px);
            box-shadow: 0px 8px 32px rgba(0,0,0,0.35);
        }}

        .prediction {{
            font-size: 42px;
            font-weight: bold;
            color: #00ff95;
            text-align:center;
        }}

        .organic {{
            color: #ffcc80;
        }}

        .confidence {{
            color: white;
            font-size: 24px;
            text-align:center;
            margin-top:10px;
        }}

        .recommend {{
            color: white;
            font-size: 20px;
            line-height: 1.8;
            margin-top: 25px;
        }}

        div.stButton > button {{
            width: 100%;
            height: 60px;
            border-radius: 18px;
            border: none;
            font-size: 20px;
            font-weight: bold;
            color: white;
            background: linear-gradient(135deg,#00c853,#00bfa5);
            box-shadow: 0px 6px 18px rgba(0,0,0,0.35);
            transition: 0.3s;
        }}

        div.stButton > button:hover {{
            transform: scale(1.03);
            background: linear-gradient(135deg,#00e676,#1de9b6);
        }}

        section[data-testid="stFileUploader"] {{
            background: rgba(255,255,255,0.12);
            padding: 20px;
            border-radius: 20px;
            border: 2px dashed rgba(255,255,255,0.5);
        }}

        .footer {{
            text-align:center;
            color:white;
            margin-top:30px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.title("♻️ Navigation")

if st.sidebar.button("🏠 Home"):
    st.session_state.page = "Home"

if st.sidebar.button("📜 Prediction History"):
    st.session_state.page = "History"

if st.sidebar.button("📊 Dashboard"):
    st.session_state.page = "Dashboard"

# ==========================================================
# HOME PAGE
# ==========================================================
if st.session_state.page == "Home":

    set_bg("home_background.jpg")

    st.markdown(
        """
        <div class="main-title">
            ♻️ Smart Waste Classification
        </div>

        <div class="sub-title">
            AI Powered Waste Detection System
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    # ======================================================
    # FILE UPLOADER
    # ======================================================
    uploaded_file = st.file_uploader(
        "📤 Upload Waste Image",
        type=["jpg", "jpeg", "png", "jfif", "webp"],
        key=f"uploader_{st.session_state.uploader_key}"
    )

    # Save uploaded file
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    # Show image
    if st.session_state.uploaded_file is not None:
        st.image(st.session_state.uploaded_file, width=350)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ======================================================
    # PREDICT BUTTON
    # ======================================================
    with col1:

        if st.button("🚀 Predict Waste Type"):

            if st.session_state.uploaded_file is None:

                st.error("⚠️ Please upload an image first.")

            else:

                with st.spinner("Analyzing Waste Image..."):

                    time.sleep(2)

                    img = image.load_img(
                        st.session_state.uploaded_file,
                        target_size=(224,224)
                    )

                    img_array = image.img_to_array(img)

                    img_array = img_array / 255.0

                    img_array = np.expand_dims(
                        img_array,
                        axis=0
                    )

                    prediction = model.predict(img_array)

                    predicted_index = np.argmax(prediction)

                    predicted_class = class_names[predicted_index]

                    confidence = round(
                        np.max(prediction) * 100,
                        2
                    )

                    st.session_state.result = predicted_class

                    st.session_state.confidence = confidence

                    st.session_state.image = st.session_state.uploaded_file

                    # ======================================================
                    # SAVE PREDICTION HISTORY
                    # ======================================================
                    new_record = {
                        "Prediction": predicted_class,
                        "Confidence (%)": confidence,
                        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    # Add to session history
                    st.session_state.history.append(new_record)

                    # Save permanently to CSV
                    history_df = pd.DataFrame(st.session_state.history)

                    history_df.to_csv(history_file, index=False)

                    # Redirect
                    st.session_state.page = "Prediction"

                    st.rerun()

    # ======================================================
    # CLEAR BUTTON
    # ======================================================
    with col2:

        if st.button("🗑️ Clear"):

            # Remove uploaded image
            st.session_state.uploaded_file = None

            # Reset uploader widget completely
            st.session_state.uploader_key += 1

            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# PREDICTION PAGE
# ==========================================================
elif st.session_state.page == "Prediction":

    set_bg("prediction_background.jpg")

    st.markdown(
        """
        <div class="main-title">
            ♻️ Prediction Result
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.image(st.session_state.image, width=450)

    predicted_class = st.session_state.result

    confidence = st.session_state.confidence

    # ======================================================
    # RECYCLABLE
    # ======================================================
    if predicted_class == "Recyclable":

        st.markdown(
            f"""
            <div class="prediction">
                ♻️ Recyclable
            </div>

            <div class="confidence">
                Confidence Score: {confidence}%
            </div>

            <div class="recommend">

                ♻️ Dispose this waste in recycling bin.<br><br>

                🌍 Recycling helps reduce pollution.<br>

                💚 Thank you for supporting clean environment.

            </div>
            """,
            unsafe_allow_html=True
        )

    # ======================================================
    # ORGANIC
    # ======================================================
    else:

        st.markdown(
            f"""
            <div class="prediction organic">
                🍂 Organic
            </div>

            <div class="confidence">
                Confidence Score: {confidence}%
            </div>

            <div class="recommend">

                🍃 Dispose in compost bin.<br><br>

                🌱 Organic recycling supports agriculture.<br>

                💚 Keep supporting sustainability.

            </div>
            """,
            unsafe_allow_html=True
        )

    st.progress(int(confidence))

    st.markdown("<br>", unsafe_allow_html=True)

    # ======================================================
    # ANALYZE AGAIN
    # ======================================================
    if st.button("🔍 Analyze Another Image"):

        st.session_state.page = "Home"

        st.session_state.uploaded_file = None

        st.session_state.uploader_key += 1

        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# HISTORY PAGE
# ==========================================================
elif st.session_state.page == "History":

    set_bg("history_background.jpg")

    st.markdown(
        """
        <div class="main-title">
            📜 Prediction History
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    history_df = pd.read_csv(history_file)

    if history_df.empty:

        st.warning("No predictions available.")

    else:

        st.dataframe(
            history_df,
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        output = BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            history_df.to_excel(
                writer,
                index=False,
                sheet_name='Prediction History'
            )

        excel_data = output.getvalue()

        st.download_button(
            label="📥 Download Excel File",
            data=excel_data,
            file_name="prediction_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# DASHBOARD PAGE
# ==========================================================
elif st.session_state.page == "Dashboard":

    set_bg("dashboard_background.jpg")

    st.markdown(
        """
        <div class="main-title">
            📊 Analytics Dashboard
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    df = pd.read_csv(history_file)

    if df.empty:

        st.warning("No data available for dashboard.")

    else:

        count_df = df["Prediction"].value_counts().reset_index()

        count_df.columns = ["Waste Type", "Count"]

        fig = px.pie(
            count_df,
            names="Waste Type",
            values="Count",
            title="Waste Classification Distribution",
            hole=0.4
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(df)

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# FOOTER
# ==========================================================
st.markdown(
    """
    <div class="footer">
        © 2026 Smart Waste Classification System | Built with ❤️
    </div>
    """,
    unsafe_allow_html=True
)