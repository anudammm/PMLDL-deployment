import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from code.models.train import CNNClassificationModel

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNClassificationModel().to(device)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "..", "..", "models", "best.pt")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

class DigitInput(BaseModel):
    image: list


@app.post("/predict/")
async def predict_digit(data: DigitInput):
    image_data = np.array(data.image).astype(np.float32).reshape(1, 1, 28, 28)
    image_tensor = torch.tensor(image_data).to(device)
    # prediction
    with torch.no_grad():
        output = model(image_tensor)
        predicted = torch.argmax(output, 1).item()  # Get predicted class
    return {"prediction": f"{predicted}"}

