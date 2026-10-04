import streamlit as st
import numpy as np
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sign Language Translator",
    page_icon="🤟",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤟 Sign Language Recognition & Translation")

st.write(
    "Convert sign language images into text and translate "
    "the recognized text into multiple languages."
)


# ============================================================
# HOW THE APPLICATION WORKS
# ============================================================

st.info(
    """
    ### 📖 How This Application Works

    **Step 1:** Upload a sign language image.

    **Step 2:** The CNN model identifies the sign.

    **Step 3:** The detected sign is converted into text.

    **Step 4:** Add multiple signs to build a word or sentence.

    **Step 5:** Select your target language.

    **Step 6:** The NLLB multilingual translation model translates
    the recognized text.

    **Example:**

    🤟 Sign Image
    → 🧠 CNN
    → 📝 Recognized Text
    → 🌍 NLLB
    → Translated Text
    """
)


# ============================================================
# LOAD CNN MODEL
# ============================================================

model_path = os.path.join(os.path.dirname(__file__), "efficientnetb0_finetuned.keras")


@st.cache_resource
def load_cnn_model():

    if not os.path.isfile(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    return load_model(model_path)


model = load_cnn_model()


# ============================================================
# CLASS NAMES
# ============================================================

class_names = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z",
    26: "del",
    27: "nothing",
    28: "space"
}


# ============================================================
# LOAD NLLB TRANSLATION MODEL
# ============================================================

@st.cache_resource
def load_translation_model():

    model_name = "facebook/nllb-200-distilled-600M"

    multi_tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    multi_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    multi_model = multi_model.to("cpu")

    return multi_tokenizer, multi_model


# ============================================================
# LANGUAGE CODES
# ============================================================

language_codes = {

    "Tamil": "tam_Taml",

    "Hindi": "hin_Deva",

    "French": "fra_Latn",

    "German": "deu_Latn",

    "Spanish": "spa_Latn"
}


# ============================================================
# TRANSLATION FUNCTION
# ============================================================

def translate_multilingual(
    text,
    target_language
):

    # Load NLLB only when translation is requested
    multi_tokenizer, multi_model = load_translation_model()

    target_code = language_codes[
        target_language
    ]

    inputs = multi_tokenizer(
        text,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        key: value.to("cpu")
        for key, value in inputs.items()
    }

    translated_tokens = multi_model.generate(

        **inputs,

        forced_bos_token_id=
        multi_tokenizer.convert_tokens_to_ids(
            target_code
        )
    )

    translated_text = multi_tokenizer.decode(
        translated_tokens[0],
        skip_special_tokens=True
    )

    return translated_text


# ============================================================
# SESSION STATE
# ============================================================

if "recognized_text" not in st.session_state:

    st.session_state.recognized_text = ""


if "last_prediction" not in st.session_state:

    st.session_state.last_prediction = ""


if "confidence" not in st.session_state:

    st.session_state.confidence = 0.0


if "translated_text" not in st.session_state:

    st.session_state.translated_text = ""


# ============================================================
# TWO COLUMN LAYOUT
# ============================================================

left_column, right_column = st.columns(2)


# ============================================================
# LEFT COLUMN - SIGN DETECTION
# ============================================================

with left_column:

    st.header("🤟 Sign Detection")

    st.write(
        "Upload a sign language image."
    )


    # --------------------------------------------------------
    # IMAGE UPLOAD
    # --------------------------------------------------------

    image_file = st.file_uploader(

        "Choose a sign language image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    if image_file is not None:

        st.image(

            image_file,

            caption="Sign Language Input",

            width=300
        )


        # ----------------------------------------------------
        # DETECT SIGN BUTTON
        # ----------------------------------------------------

        if st.button(

            "🤟 Detect Sign",

            use_container_width=True
        ):

            try:

                # ==================================================
                # UPLOAD IMAGE PROCESSING
                # ==================================================

                img = load_img(

                    image_file,

                    target_size=(
                        224,
                        224
                    )
                )


                img_array = img_to_array(
                    img
                )


                img_array = np.expand_dims(

                    img_array,

                    axis=0
                )


                # ==================================================
                # CNN PREDICTION
                # ==================================================

                prediction = model.predict(

                    img_array,

                    verbose=0
                )


                # Get predicted class index

                predicted_index = np.argmax(
                    prediction
                )


                # Get predicted sign

                predicted_class = class_names[
                    predicted_index
                ]


                # Get confidence

                confidence = float(

                    prediction[
                        0
                    ][
                        predicted_index
                    ]

                ) * 100


                # ==================================================
                # SAVE PREDICTION
                # ==================================================

                st.session_state.last_prediction = (
                    predicted_class
                )

                st.session_state.confidence = (
                    confidence
                )


                # ==================================================
                # BUILD RECOGNIZED TEXT
                # ==================================================

                if predicted_class == "space":

                    st.session_state.recognized_text += " "


                elif predicted_class == "del":

                    st.session_state.recognized_text = (

                        st.session_state.recognized_text[:-1]

                    )


                elif predicted_class == "nothing":

                    pass


                else:

                    st.session_state.recognized_text += (

                        predicted_class

                    )


                # ==================================================
                # CONFIDENCE MESSAGE
                # ==================================================

                if confidence < 60:

                    st.warning(

                        f"⚠️ Low confidence: "
                        f"{confidence:.2f}%. "
                        "Try uploading a clearer sign image."
                    )

                else:

                    st.success(

                        f"Sign detected successfully: "
                        f"{predicted_class}"
                    )


            except Exception as e:

                st.error(

                    f"Error while detecting sign: {e}"
                )


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    if st.session_state.last_prediction:

        st.divider()

        st.subheader(
            "🔍 Prediction Result"
        )


        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.metric(

                "Detected Sign",

                st.session_state.last_prediction
            )


        with result_col2:

            st.metric(

                "Confidence",

                f"{st.session_state.confidence:.2f}%"
            )


    # ========================================================
    # RECOGNIZED TEXT
    # ========================================================

    st.divider()

    st.subheader(
        "📝 Recognized Text"
    )


    if st.session_state.recognized_text:

        st.success(
            st.session_state.recognized_text
        )

    else:

        st.info(
            "No signs detected yet. "
            "Add a sign image to begin."
        )


    # ========================================================
    # TEXT CONTROLS
    # ========================================================

    button_col1, button_col2 = st.columns(2)


    with button_col1:

        if st.button(

            "↩️ Delete Last Letter",

            use_container_width=True
        ):

            if st.session_state.recognized_text:

                st.session_state.recognized_text = (

                    st.session_state.recognized_text[:-1]

                )

                st.rerun()


    with button_col2:

        if st.button(

            "🗑️ Clear Text",

            use_container_width=True
        ):

            st.session_state.recognized_text = ""

            st.session_state.last_prediction = ""

            st.session_state.confidence = 0.0

            st.session_state.translated_text = ""

            st.rerun()


# ============================================================
# RIGHT COLUMN - TRANSLATION
# ============================================================

with right_column:

    st.header("🌐 Translation")

    st.write(
        "Translate the recognized sign language text "
        "into your selected language."
    )


    # ========================================================
    # DETECTED TEXT
    # ========================================================

    st.subheader(
        "📝 Detected Text"
    )


    if st.session_state.recognized_text:

        st.info(
            st.session_state.recognized_text
        )

    else:

        st.info(
            "Your recognized text will appear here."
        )


    # ========================================================
    # TARGET LANGUAGE
    # ========================================================

    target_language = st.selectbox(

        "🌍 Choose Target Language",

        list(
            language_codes.keys()
        )
    )


    # ========================================================
    # TRANSLATE BUTTON
    # ========================================================

    if st.button(

        "🌍 Translate",

        use_container_width=True
    ):

        if st.session_state.recognized_text.strip():

            try:

                with st.spinner(
                    "Translating using NLLB..."
                ):

                    translated_text = (
                        translate_multilingual(

                            st.session_state.recognized_text,

                            target_language
                        )
                    )


                st.session_state.translated_text = (

                    translated_text

                )


            except Exception as e:

                st.error(

                    f"Translation error: {e}"
                )


        else:

            st.warning(
                "Please add signs first."
            )


    # ========================================================
    # TRANSLATED OUTPUT
    # ========================================================

    if st.session_state.translated_text:

        st.divider()

        st.subheader(

            f"🌍 {target_language} Translation"
        )


        st.success(

            st.session_state.translated_text
        )


    # ========================================================
    # TRANSLATION MODEL INFORMATION
    # ========================================================

    st.divider()


    with st.expander(
        "🧠 About the Translation Model"
    ):

        st.write(
            """
            **Model:** NLLB-200 Distilled 600M

            **NLLB:** No Language Left Behind

            NLLB is a multilingual neural machine translation
            model developed to translate text between many
            languages.

            In this application, the recognized sign language
            text is given as input to NLLB and translated into
            the selected target language.
            """
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()


st.header(
    "🧠 Model Information"
)


model_col1, model_col2, model_col3 = st.columns(3)


with model_col1:

    st.info(
        """
        **Sign Recognition**

        Model: EfficientNetB0

        Input Size: 224 × 224

        Output Classes: 29
        """
    )


with model_col2:

    st.info(
        """
        **Translation**

        Model: NLLB-200

        Model Size: Distilled 600M

        Task: Multilingual Translation
        """
    )


with model_col3:

    st.info(
        """
        **Recognition Classes**

        A – Z

        Space

        Delete

        Nothing
        """
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()


with st.expander(
    "📊 Model Performance"
):

    st.write(
        """
        ### EfficientNetB0 Sign Recognition

        The EfficientNetB0 model was trained to recognize
        **29 sign language classes**, including A–Z, space,
        delete, and nothing.

        **Best Validation Accuracy: approximately 92.95%**

        The model uses a pretrained EfficientNetB0 backbone
        with a custom classification layer and softmax output.
        """
    )


# ============================================================
# PROJECT PIPELINE
# ============================================================

st.divider()


st.subheader(
    "🔄 Application Pipeline"
)


st.write(
    """
    📤 **Image Upload**

    ↓

    🧹 **Image Preprocessing**

    ↓

    🧠 **EfficientNetB0 Sign Recognition**

    ↓

    📝 **Sign → Text**

    ↓

    🌍 **NLLB Multilingual Translation**

    ↓

    ✅ **Translated Text**
    """
)