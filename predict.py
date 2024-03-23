from keras.models import load_model
from solution import solution
import tensorflow as tf
import numpy as np

class_labels = [
    'Melanocytic nevi (non-cancerous)',
    'Melanoma (cancerous)',
    'Benign keratosis-like lesions (non-cancerous)',
    'Basal cell carcinoma (cancerous)',
    'Actinic keratoses (non-cancerous)',
    'Vascular lesions (non-cancerous)',
    'Dermatofibroma (non-cancerous)'
]

# for bytes
def prepare_image(image, image_size):
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.cast(image, tf.float32)
    image /= 255.0
    image = tf.image.resize(image, [image_size, image_size])

    image = np.expand_dims(image, axis=0)

    return image

def classify_using_bytes(image_bytes, model_path, image_size):
    model = load_model(model_path, compile=False)
    model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    prediction = model.predict(prepare_image(image_bytes, image_size))
    index = np.argmax(prediction, axis=1)[0]
    sol = solution[index]

    class_name = class_labels[index]
    confidence_score = prediction[0][index]

    result = {
        'class' : class_name,
        'score' : f'{confidence_score*100:02.2f}%'
    }

    result.update(sol)

    return result 


if __name__ == '__main__':
    pth = "download.jpeg"
    with open(pth, "rb") as file:
        print(classify_using_bytes(file.read(), 'weights.h5', 28))

