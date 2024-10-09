from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import utils
import matplotlib.pyplot as plt
import uvicorn
from PIL import Image
import io

app = FastAPI()

model = utils.setup(model_path="./model/MNIST_ResNet18.pt")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model to define the structure of incoming data


class Item(BaseModel):
    data: list[int]


@app.post("/send-data")
async def receive_data(item: Item):
    received_data = item.data
    image = utils.transform(received_data)
    output = utils.predict(model, image)
    probability, prediction = utils.get_output_dict(output)

    return {
        "message": "Data received successfully",
        "received_data": received_data,
        "result": prediction,
        "probability": probability,
    }


# Run the application


if __name__ == "__main__":
    print("heeereee")
    uvicorn.run(app, host="localhost", port=4000)
