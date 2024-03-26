from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2

app = FastAPI()

origins = [
    "exp://192.168.1.4",
    "exp://192.168.1.4:8081",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


MODEL = tf.keras.models.load_model("../saved_models/1")
CLASS_NAMES = ["Black-blight","Blister-blight","Canker","Horse-hair-blight","brown blight","healthy"]

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


def resize_image(image,target_size = (512,512)):
    resized_image = cv2.resize(image,target_size)
    return resized_image

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    print("Image Shape:", image.shape)
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    orginal_image = read_file_as_image(await file.read())
    print("File Name:", file.filename)
    print("Content Type:", file.content_type)
    print("File Size (bytes):", len(await file.read()))

    resized_image = resize_image(orginal_image)
    print("Resized Image shape: ",resized_image.shape)

    img_batch = np.expand_dims(resized_image, 0)

    prediction = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    recommendation = RECOMMENDATIONS.get(predicted_class, "No recommendation found")
    return{
        'diseaseName': predicted_class,
        'confidence': float(confidence),
        'recommendation':recommendation
    }




if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)