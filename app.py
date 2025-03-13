import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load the pre-trained emotion detection model
pipe_lr = joblib.load(open("text_emotion.pkl", "rb"))

# Dictionary mapping emotions to emojis
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", 
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# Predict emotion from input text
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

# Get the prediction probabilities
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    # Add a sidebar for additional features or explanation
    st.sidebar.header("About")
    st.sidebar.write("""
        This application detects emotions in the text you provide. 
        It uses a machine learning model trained to classify emotions like anger, happiness, sadness, etc.
        Simply type or upload some text and get the detected emotion along with the confidence level.
    """)

    # Choose between text input and file upload
    input_mode = st.radio("Choose Input Mode", ("Type Text", "Upload Text File"))
    
    if input_mode == "Type Text":
        with st.form(key='my_form'):
            raw_text = st.text_area("Type Here", placeholder="Type your text here to detect emotion")
            submit_text = st.form_submit_button(label='Submit')

    elif input_mode == "Upload Text File":
        uploaded_file = st.file_uploader("Choose a file", type="txt")
        if uploaded_file is not None:
            raw_text = uploaded_file.read().decode("utf-8")
            st.text_area("Uploaded Text", raw_text, height=200)
            submit_text = True
        else:
            submit_text = False

    # Process text when the form is submitted
    if submit_text:
        col1, col2 = st.columns(2)

        # Predict emotion and probabilities
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "😶")  # Default emoji if emotion not found
            st.write(f"Emotion: {prediction} {emoji_icon}")
            st.write(f"Confidence: {np.max(probability) * 100:.2f}%")

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["Emotions", "Probability"]

            # Display a bar chart of prediction probabilities
            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='Emotions', y='Probability', color='Emotions'
            ).properties(title="Emotion Prediction Probabilities")
            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
