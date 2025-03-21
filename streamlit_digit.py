import numpy as np
import matplotlib.pyplot as plt
import joblib
from PIL import Image
import streamlit as st

from streamlit_drawable_canvas import st_canvas
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier

#Loading the model. Url should change to correct folder of model if deployed
model = joblib.load(r'C:\Users\nexuz\Desktop\Peter\Skola\05_ML\programmering\mina_repositorys\05_ml\models\final_model\voting_soft.joblib')
#Loading scaler. Url should change to correct folder of scaler if deployed
scaler = joblib.load(r'C:\Users\nexuz\Desktop\Peter\Skola\05_ML\programmering\mina_repositorys\05_ml\scaler\standard_scaler.joblib')

st.title('Welcome to this amazing world of number prediction with a white-box model!')
st.header('Just draw a number and watch the magic unfold!')

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    update_streamlit=True,
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype(np.uint8))
    img = img.convert('L')  # Converting the image to grey
    img = img.resize((28, 28))  
    img = np.array(img)

    if np.any(img < 255):  # Check if anything is drawn
        img = np.invert(img)  # Invert color

        # Preeprocessing
        img_2d = img.flatten().reshape(1, -1)  # 28x28 -> (1, 784)
        scaled_img = scaler.transform(img_2d) #scaling with loaded scaler.

        # Predictioon
        pred = model.predict(scaled_img)
        st.write(f"Predicted digit: {pred[0]}")
    else:
        st.write("Draw a digit to get a prediction!")