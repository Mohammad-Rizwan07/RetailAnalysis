


###-----IMPORTS-----###
from ultralytics import YOLO
import os
import cv2



###-----SETUP-----###
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'ModelTraining/runs/detect/train/weights/best.pt')
model = YOLO(model_path)



###-----INFERENCE-----###
image_path = os.path.join(base_dir, 'Data/Vitaly.Okhonya_2020_11_26_10_33_39_1606376019160.jpg')
results = model(image_path)



###-----SAVE RESULTS-----###
result_save_path = os.path.join(base_dir, 'Results/result.jpg')
results[0].save(result_save_path)

res_plotted = results[0].plot()
result_cv2_path = os.path.join(base_dir, 'Results/result_cv2.jpg')
cv2.imwrite(result_cv2_path, res_plotted)


