from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import utils
import matplotlib.pyplot as plt
import uvicorn
from PIL import Image
import io

app = FastAPI()

model = utils.setup(model_path="./model/Numbers_recegnition_model.pt")

# FastAPI.
# Allow CORS (Cross-Origin Resource Sharing)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# app.add_api_route
# Pydantic model to define the structure of incoming data


class Item(BaseModel):
    data: list[int]


# Define route to receive data from React frontend


@app.post("/send-data")
# async def receive_data(file: UploadFile = File(...)):
#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents)).convert('L')

#     # Preprocess the image
#     image_tensor = utils.transform(image)

#     # Get prediction
#     output = utils.predict(model, image_tensor)
#     prediction = utils.get_output_dict(output)

#     return {"message": "Data received successfully", "result": prediction}


async def receive_data(item: Item):
    received_data = item.data
    image = utils.transform(received_data)
    # print(image)
    # plt.imshow(image.view(28, 28), cmap="gray")
    # plt.show()
    output = utils.predict(model, image)
    # Send a response back to the React frontend
    probability, prediction = utils.get_output_dict(output)

    print(prediction)
    return {
        "message": "Data received successfully",
        "received_data": received_data,
        "result": prediction,
        "probability": probability
    }


# Run the application


if __name__ == "__main__":
    print("heeereee")
    uvicorn.run(app, host="localhost", port=4000)
