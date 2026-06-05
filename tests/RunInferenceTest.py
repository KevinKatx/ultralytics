from ultralytics import YOLO
import torch
print("Is CUDA Available?")
print(torch.cuda.is_available())


model_path = input("Input Model Path:")
test_dataset_path = input("Input Test Dataset Path:")
output_path = input("Indicate Path for output:")
testName = input("Indicate Name of Test:")


print(f"Model path:{model_path}")
print(f"Dataset path:{test_dataset_path}")
print(f"Output path:{output_path}")
print(f"Folder Name:{testName}")

model = YOLO(model_path)
model.predict(source=rf"{test_dataset_path}", save=True, project=rf"{output_path}", name=testName )

