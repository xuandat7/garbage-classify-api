from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model('models/model.resnet50.h5')
output_class = ["cardboard", "glass", "metal","paper", "plastic", "trash"]
    
async def predict(new_image_path):
    try:
        test_image = image.load_img(new_image_path, target_size=(224, 224))
        test_image = image.img_to_array(test_image) / 255
        test_image = np.expand_dims(test_image, axis=0)

        predicted_array = model.predict(test_image)
        predicted_value = output_class[np.argmax(predicted_array)]
        predicted_accuracy = round(np.max(predicted_array) * 100, 2)

        return predicted_value, predicted_accuracy
    except Exception as e:
        return f"Error processing image: {str(e)}", 0