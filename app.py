import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
existing_model = tf.keras.models.load_model('aaaaa.hdf5', compile = False)
existing_model_cancer = tf.keras.models.load_model('ensemble_model.hdf5', compile = False)

# Define your class names in the same order as your model's output
class_names = ["Acne", "Eczema", "Melanoma", "Normal", "Psoriasis", "Tinea", "vitiligo"]  # Replace with your actual class names

class_names_cancer = ["actinic keratosis", "basal cell carcinoma", "benign keratosis", "dermatofibroma", "melanoma", "melanocytic nevus", "vascular lesion"]

disease_info = {
    "Acne": {
        "Description": "Acne is a common skin condition that occurs when hair follicles are clogged with oil and dead skin cells. It often causes pimples and can vary in severity.",
        "Causes": "Causes of acne include hormonal changes, excess oil production, bacteria, and certain medications.",
        "Treatment": "Treatment options for acne may include topical creams, oral medications, and lifestyle changes.",
    },
    "Ezcema": {
        "Description": "Acne is a common skin condition that occurs when hair follicles are clogged with oil and dead skin cells. It often causes pimples and can vary in severity.",
        "Causes": "Causes of acne include hormonal changes, excess oil production, bacteria, and certain medications.",
        "Treatment": "Treatment options for acne may include topical creams, oral medications, and lifestyle changes.",
    },
    "Melanoma": {
        "Description": "Melanoma is a type of skin cancer that originates in the melanocytes, the pigment-producing cells in the skin. It is considered the most dangerous form of skin cancer because it can rapidly spread to other parts of the body if not detected and treated early. Melanoma often appears as an unusual mole or pigmented skin lesion that can change in size, shape, or color over time.",
        "Causes": "The primary cause of melanoma is exposure to ultraviolet (UV) radiation from sunlight or artificial sources like tanning beds. Individuals with a history of excessive sun exposure, sunburns, a family history of melanoma, numerous moles, fair skin, and a weakened immune system are at higher risk. UV radiation damages DNA in skin cells, leading to the development of cancerous cells.",
        "Treatment": "Treatment options for melanoma depend on the stage of the cancer and may include:Surgery, Immunotherapy, Targeted Therapy, Chemotherapy, Radiation Therapy, Clinical Trials",
    },
    "Normal":{
        "Description": "Normal",
        "Causes": "Normal",
        "Treatment": "Normal",
    },
    "Psoriasis":{
        "Description": "Psoriasis is a chronic autoimmune skin condition characterized by the rapid buildup of skin cells, leading to the formation of red, scaly patches on the skin's surface. It can affect various parts of the body and often has periods of exacerbation and remission.",
        "Causes": "The exact cause of psoriasis is not fully understood, but it is thought to involve a combination of genetic, immune system, and environmental factors. Triggers can include stress, infections, certain medications, and injuries to the skin.",
        "Treatment": "Treatment for psoriasis aims to reduce inflammation, slow down skin cell growth, and alleviate symptoms. Common treatments include topical corticosteroids, phototherapy, oral medications, and biologic drugs. Lifestyle modifications, such as managing stress and avoiding triggers, can also help manage the condition.",
    },
    "Tinea":{
        "Description": "Tinea, commonly known as ringworm, is a contagious fungal infection that affects the skin, scalp, or nails. It often presents as a red, itchy rash with a ring-like appearance. Despite its name, ringworm is not caused by a worm but by various types of fungi.",
        "Causes": "Tinea is caused by different species of fungi known as dermatophytes. These fungi thrive in warm and humid environments and can be transmitted through direct contact with an infected person or contaminated objects.",
        "Treatment": "Treatment for tinea typically involves antifungal medications, either topical (creams or ointments) or oral (pills). Good hygiene practices, such as keeping the affected area clean and dry, are important for preventing the spread of the infection.",
    },
    "vitiligo":{
        "Description": "Vitiligo is a long-term skin condition characterized by the loss of skin pigment, resulting in white patches on the skin. It occurs when melanocytes, the cells responsible for producing pigment, are destroyed or stop functioning.",
        "Causes": "The exact cause of vitiligo is not fully understood, but it is believed to involve a combination of genetic and autoimmune factors. It is not contagious, and it can occur at any age.",
        "Treatment": "While there is no cure for vitiligo, various treatments can help manage the condition. These treatments may include topical corticosteroids, topical calcineurin inhibitors, narrowband ultraviolet B (NB-UVB) therapy, and surgical options such as skin grafting. Treatment choices depend on the extent and location of the white patches and should be discussed with a dermatologist.",
    },
}

disease_info_cancer = {
"actinic keratosis":{
        "Description": "Actinic keratosis, also known as solar keratosis or senile keratosis, is a precancerous skin condition caused by prolonged sun exposure. It typically appears as dry, scaly patches or rough, crusty growths on the skin. If left untreated, actinic keratosis can develop into skin cancer"
},
"basal cell carcinoma":{
        "Description": "Basal cell carcinoma (BCC) is the most common form of skin cancer. It usually presents as a small, pearly bump or a pinkish patch on the skin. BCC tends to grow slowly and rarely metastasizes, but it can cause local damage if not treated promptly."
},
"benign keratosis":{
        "Description": "Benign keratosis, also known as seborrheic keratosis or senile wart, is a non-cancerous skin growth. It appears as brown or black warty lesions that are typically painless and often found on the face, chest, back, or other areas of the body. These growths are generally harmless but can be cosmetically bothersome."
},
"dermatofibroma":{
        "Description": "Dermatofibroma is a benign skin tumor that commonly occurs on the legs. It presents as a firm, raised nodule with a reddish-brown coloration. Dermatofibromas are usually harmless but can be itchy or tender."
},
"melanoma":{
        "Description": "Melanoma is a serious and potentially deadly form of skin cancer that originates in melanocytes, the cells responsible for producing pigment in the skin. It often appears as an irregularly shaped mole with uneven coloration, jagged borders, and can grow rapidly. Early detection and treatment are crucial for a better prognosis."
},
"melanocytic nevus":{
        "Description": "A melanocytic nevus, commonly referred to as a mole, is a benign growth of pigment-producing cells (melanocytes). Moles can vary in size, shape, and color and are usually harmless. However, changes in size, shape, or color should be monitored, as they may indicate the development of melanoma."
},
"vascular lesion":{
        "Description": "Vascular lesions encompass a group of skin conditions characterized by abnormalities in blood vessels. These can include birthmarks (e.g., port-wine stains), hemangiomas, telangiectasias (e.g., spider veins), and other vascular malformations. Vascular lesions can be benign or may require medical attention depending on their type and location."
},
}


disease_videos = {
    "Acne": "",
    "Eczema": "https://www.youtube.com/watch?v=VIDEO_ID_FOR_ECZEMA",
    "Melanoma": "https://www.youtube.com/watch?v=VIDEO_ID_FOR_MELANOMA",
    "Normal": "https://www.youtube.com/watch?v=VIDEO_ID_FOR_NORMAL",
    "Psoriasis": "https://www.youtube.com/watch?v=VIDEO_ID_FOR_PSORIASIS",
    "Tinea": "https://www.youtube.com/watch?v=VIDEO_ID_FOR_TINEA",
    "vitiligo": "https://youtu.be/Zz35mjTdse4?si=Fc4BFYoDTRH22gmc",
}
def predict_skin_disease(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Adjust target_size as needed
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    predictions = existing_model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

def predict_skin_cancer_disease(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Adjust target_size as needed
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    predictions = existing_model_cancer.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name_cancer = class_names_cancer[predicted_class_index]
    return predicted_class_name_cancer

@app.route('/',methods=["GET", "POST"])
def index():
  return render_template('index.html')

@app.route('/contact',methods=["GET", "POST"])
def contact():
  return render_template('contact.html')

@app.route('/client', methods=["GET", "POST"])
def client():
  return render_template('client.html')

@app.route('/medicine',methods=["GET", "POST"])
def medicine():
  return render_template('medicine.html')

@app.route('/test',methods=["GET", "POST"])
def test():
  return render_template('test.html')

@app.route('/common', methods=['GET', 'POST'])
def common():
    predicted_class = None
    disease_description = None
    disease_causes = None
    disease_treatment = None
    disease_video_url = None
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        # Check if the file has a filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file:
            # Save the uploaded file to a temporary location
            temp_file_path = "temp.jpg"
            file.save(temp_file_path)
            # Get the predicted class
            predicted_class = predict_skin_disease(temp_file_path)
            # Fetch disease information and video URL based on the predicted class
            if predicted_class in disease_info:
                disease_description = disease_info[predicted_class]["Description"]
                disease_causes = disease_info[predicted_class]["Causes"]
                disease_treatment = disease_info[predicted_class]["Treatment"]
                if predicted_class in disease_videos:
                    disease_video_url = disease_videos[predicted_class]
    return render_template('common.html', predicted_class=predicted_class, disease_description=disease_description,
                           disease_causes=disease_causes, disease_treatment=disease_treatment,
                           disease_video_url=disease_video_url)


@app.route('/cancer', methods=['GET', 'POST'])
def cancer():
    predicted_class_cancer = None
    disease_description_cancer = None
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        # Check if the file has a filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file:
            # Save the uploaded file to a temporary location
            temp_file_path = "temp1.jpg"
            file.save(temp_file_path)
            # Get the predicted class
            predicted_class_cancer = predict_skin_cancer_disease(temp_file_path)
            # Fetch disease information and video URL based on the predicted class
            if predicted_class_cancer in disease_info_cancer:
                disease_description_cancer = disease_info_cancer[predicted_class_cancer]["Description"]

    return render_template('cancer.html', predicted_class_cancer=predicted_class_cancer, disease_description=disease_description_cancer)

if __name__ == "__main__":
    app.run(debug=True)


