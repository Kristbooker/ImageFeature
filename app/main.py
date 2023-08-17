import base64
import cv2
from fastapi import FastAPI, HTTPException,Request
import numpy as np

app = FastAPI()

@app.get("/")
def root():
    return {"message": "This is my api"}

#HOG
@app.get("/api/genhog")
async def gen_hog(data:Request):
    try:
        json=await data.json()
        img_data=json["img_data"]#key name
        #split to remove data:image/jpeg;base64, 
        split_img_data = img_data.split(',', 1)[1]
        # Decode base64 image
        image_bytes = base64.b64decode(split_img_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        img_gray = cv2.imdecode((image_array), cv2.IMREAD_GRAYSCALE)
         
        resized = cv2.resize(img_gray, (128, 128),cv2.INTER_AREA)
        resized_height, resized_width = resized.shape

        win_size = (resized_width,resized_height)
        cell_size = (8, 8)
        block_size = (16, 16)
        block_stride = (8, 8)
        num_bins = 9
        
        # Set the parameters of the HOG descriptor using the variables defined above
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
        
        # Compute the HOG Descriptor for the gray scale image
        hog_descriptor = hog.compute(resized)
        # HOG descriptor to a list
        hog_vector = hog_descriptor.flatten().tolist()
        return {"HOG VECTOR ": hog_vector}
        
    except Exception as e:
        print("Error:", str(e)) 
        raise HTTPException(status_code=500, detail="Error")

    


