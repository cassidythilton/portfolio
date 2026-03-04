import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pandas as pd
from scipy.spatial import distance
from PIL import Image


#terminal command: 
#    streamlit run str.py

def calculate_similarity(row, user_input):
    row_values = row[['n_citi', 'bed', 'bath', 'sqft']].values
    return distance.euclidean(row_values, user_input)

df=pd.read_csv('src/homePrices.csv')
model = load_model('src/housePrices.h5')

#st.set_page_config(page_title="Home Price Predictor", layout='wide')
#st.title('Home Price Predictor')
@st.cache_data
def load_image(image_path):
    from PIL import Image
    return Image.open(image_path)

image = load_image('src/reHeader.png')
st.image(image, caption='Header', use_column_width=True)

#st.image("src/reHeader.png", use_column_width=True, output_format='png')


# Input text boxes for tabular data
citiM, bM, bathM, sqftM, priceM = 414, 12, 36.0, 17667, 1000000

n_citi = st.number_input('Enter the City code:', min_value=0.0)
bed = st.number_input('Enter the total number of Beds:', min_value=0.0)
bath = st.number_input('Enter the total number of Baths:', min_value=0.0)
sqft = st.number_input('Enter the total Square Footage:', min_value=0.0)

# Upload image
uploaded_image = st.file_uploader("Upload an image of the home", type=["jpg", "png", "jpeg"])

# Prepare image
if uploaded_image is not None:
    image_sample = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), -1)

    # Convert from BGR to RGB (OpenCV loads images in BGR)
    image_sample_rgb = cv2.cvtColor(image_sample, cv2.COLOR_BGR2RGB)

    # Display the original image
    st.image(image_sample_rgb, caption='Uploaded Image', use_column_width=True)

    # Further processing (resize etc.)
    sample_resized = cv2.resize(image_sample, (64, 64)) / 255.0

    # Prepare tabular data
    X1_final = np.array([n_citi, bed, bath, sqft], dtype='float32')

    # Generate prediction
    if st.button('Predict'):
        y_pred = model.predict([np.reshape(X1_final, (1, 4)), np.reshape(sample_resized, (1, 64, 64, 3))])
        st.write(f"Predicted Price: ${int(y_pred * priceM):,}")


        # Create a new column in the DataFrame to store similarity values
        user_input = np.array([n_citi, bed, bath, sqft])
        df['similarity'] = df.apply(lambda row: calculate_similarity(row, user_input), axis=1)

        # Sort by similarity and select the top 4 homes
        df_sorted = df.sort_values(by='similarity', ascending=True).head(4)

        st.write('### Top 4 Comparable Homes')
        for index, row in df_sorted.iterrows():
            # Display basic info
            st.write(f"Home {index+1}")
            st.write(f"Price: ${row['price']:,}, Beds: {row['bed']}, Baths: {row['bath']}, Sqft: {row['sqft']:,}")
            
            # Display image
            image_path = f"img/{row['image_id']}.jpg"
            image = cv2.imread(image_path)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Image {row['image_id']}", use_column_width=True)