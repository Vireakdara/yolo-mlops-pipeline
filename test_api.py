import requests

url = "http://localhost:8080/detect"
image_path = r"E:\\Dataset\\coco128\\images\\train2017\\000000000110.jpg"

with open(image_path, "rb") as f:
    response = requests.post(url, files={"file": f})

print(response.json())