from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

BUCKET_NAME = "tea-disease-99"
class_names = ["Black-blight", "Blister-blight", "Canker", "Horse-hair-blight", "brown blight", "healthy"]

model = None

RECOMMENDATIONS = {
    "Blister-blight": "Planting suitable clones : Natural resistance inherent by the plant..."
                      "Management of shade trees – Thinning / Lopping...."
                      "Chemical control method:  - Spraying fungicides - Copper fungicides - Protective  "
                      "- Systemic fungicides - Curative..."
                      "Nurseries - spray copper fungicides every four days 120g in 45 l water...."
                      "Tea recovering from pruning - Spray every 4 or 5 days, 420g in 170 l of water per"
                      "hectare using knapsack sprayers or 420 g in 30-45 l per hectare using "
                      "mist-blowers up to the time of tipping..."
                      "Tea in plucking - Spray once every 7-10 days depending on plucking round"
                      "280-420g in 170 l of water per hectare using knapsack sprayers or"
                      "280g-420g in 30 -45 l water per hectare using mist-blowers.",

    "Black-blight": "Chemical control method: - Spraying fungicides  - Copper fungicides - Protective  "
                    "- Systemic fungicides - Curative....."
                    "In heavily shaded nurseries during heavy rainfall, thin out shade and apply a 50% w/w copper "
                    "fungicide spray (120g in 45l water) with knapsack sprayers at leaf spotting onset. Repeat after "
                    "two weeks if rain persists. In new clearings, spray when stem infection appears using the same "
                    "copper fungicide dilution but with approximately 430l water per hectare for thorough coverage. ",

    "brown blight": "No special control measures are recommended... "
                    "When outbreaks occur Copper fungicides could be used.."
                    "Control measures are not necessary in these cases. With regular spraying done for "
                    "blister bligh this disease do not pose any problem any more.",

    "Canker": "* Removal of affected branches at pruning...Avoid planting tea in poor soil areas..."
              "*Use vigorous plants with developed root system  for planting...Adopt proper soil conservation measures "
              "* Avoid planting susceptible cultivars (TRI 2023, TRI 2026) in risky areas ..."
              "Avoid mechanical damage to stems"
              "Thatch soil during dry weather",

    "Horse-hair-blight": "Remove a and destroy all crop debris from around plants"
                         "Prune out infected or dead branches from the plant canopy",

    "healthy": "This leaf is in excellent health, showing no signs of disease."
}

def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/teadiseases.h5",
            "/tmp/teadiseases.h5",
        )
        model = tf.keras.models.load_model("/tmp/teadiseases.h5")

    image = request.files["file"]
    image = np.array(Image.open(image).convert("RGB").resize((512, 512)))
    image = image / 255
    img_array = tf.expand_dims(image, 0)

    predictions = model.predict(img_array)
    print(predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    recommendation = RECOMMENDATIONS.get(predicted_class,"No recommendation found")

    return {"diseaseName": predicted_class, "confidence": confidence, "recommendation": recommendation}
