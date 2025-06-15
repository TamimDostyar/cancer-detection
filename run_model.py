from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

def run_model(img_path):
    IMG_SIZE = (224, 224)
    model = tf.keras.models.load_model("/app/models/my_cancer_cnn_model.h5")
    class_labels = {
        "lung_aca": "Lung Adenocarcinoma",
        "lung_n": "Lung Benign Tissue (Normal)",
        "lung_scc": "Lung Squamous Cell Carcinoma",
        "colon_aca": "Colon Adenocarcinoma",
        "colon_n": "Colon Benign Tissue (Normal)"
    }
    print("Class labels:", class_labels)
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_key = list(class_labels.keys())[np.argmax(score)]
    confidence = (100 * np.max(score))
    return predicted_key, confidence