import os
from flask import current_app

import numpy as np
import tensorflow as tf
from scipy.stats import mode

from agents.chatAgent.agentHandler import runAgent

tf.config.set_visible_devices([], 'GPU')

conversationHistory = []

def handleChatService(message, image):
    """
    Handles saving of uploaded image and text.
    Returns a dict with upload details.
    """
    image_path = None

    print(message)
    
    try:

        image_path = imageStoreService( image )

        if image_path is not None :
            diseaseFound = diseaseDetectService( 'leaf' , 'turmeric' , image_path )

        if( message is None or message == "" ) and image_path is None:

            return {
                "status": 200,
                "message": " Can't able to process,can you give some detail",
                "DiseaseName" : None,
                "image_path": None
                }
        
        elif (message is not None or message != "") and image_path is not None:

            message = f"diseaseName :{diseaseFound} \n\n"+message
            
            resp = runAgent(conversationHistory=conversationHistory , message = message)

            return {
                "status" : 200,
                "message" : resp,
                "DiseaseName" : diseaseFound,
                "image_path" : image_path
            }

        elif (message is None or message == ""):

            message = f"Explain about this disease \n diseaseName : { diseaseFound}"

            resp = runAgent(conversationHistory=conversationHistory , message = message)

            return {
                "status" : 200,
                "message" : resp,
                "DiseaseName" : diseaseFound ,
                "image_path" : image_path
            }

        elif image_path is None :

            message = " Explain this question with previous context \n\n"+message

            resp = runAgent(conversationHistory=conversationHistory , message = message)

            print(resp)
            
            return {
                "status" : 200,
                "message" : resp,
                "DiseaseName" : None,
                "image_path" : None
            }



    except Exception as e :

        print(f"Error Occured : {e}")
        
        return {
            "status" : 500,
            'message' : e,
            "image_path" : None
        }


def imageStoreService( image ):

    image_path = None

    try:

        if image :

            upload_folder = current_app.config["UPLOAD_FOLDER"]
            os.makedirs( upload_folder , exist_ok = True )

            image_path = os.path.join( upload_folder , image.filename )

            image.save( image_path )

    
        return image_path

    except Exception as e:

        print(f"Exception occured at chatService (savingImage) {e}")
        return None

def diseaseDetectService(img_type, crop_name, file_path):

    model1 = tf.keras.models.load_model(f'../models/inception/{img_type}/{crop_name}.h5')  # Inception-based
    model2 = tf.keras.models.load_model(f'../models/mobilenet/{img_type}/{crop_name}.h5')  # MobileNet-based
    model3 = tf.keras.models.load_model(f'../models/resnet/{img_type}/{crop_name}.h5')  # ResNet-based
    
    #print("Model Loaded successfully")
    # Define the labels
    ensem_label = {
        'apple-leaf': ['apple_scab', 'apple_blackrot', 'apple_Cedar_rust', 'apple_healthy'],
        'apple-fruit': ['apple_blotch', 'apple_healthy', 'apple_blackrot', 'apple_scab'],
        'turmeric-leaf':['leaf_blotch','healthy','leaf_spot']
    }
    labels = ensem_label[crop_name + "-" + img_type]

    def preprocess_image(image_path, target_size):
        """Preprocess image for model prediction."""
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array /= 255.0  # Normalize
        return image_array

    def ensemble_predictions(models, image_path):
        """Get ensemble predictions from multiple models."""
        # Define target sizes for each model
        model_input_sizes = {
            'inception': (299, 299),  # Inception model typically expects 299x299
            'mobilenet': (224, 224),  # MobileNet model typically expects 224x224
            'resnet': (224, 224),     # ResNet model also typically expects 224x224
        }

        # Preprocess image for each model
        preprocessed_images = {
            'inception': preprocess_image(image_path, model_input_sizes['inception']),
            'mobilenet': preprocess_image(image_path, model_input_sizes['mobilenet']),
            'resnet': preprocess_image(image_path, model_input_sizes['resnet'])
        }

        # Get predictions from each model
        predictions = []
        individual_predictions = {}
        for model, model_name in zip(models, ['Inception', 'MobileNet', 'ResNet']):
            pred = model.predict(preprocessed_images[model_name.lower()])
            confidence_score = np.max(pred)  # Get the confidence score (highest probability)
            predicted_label_index = np.argmax(pred)  # Get the predicted label index
            predicted_label = labels[predicted_label_index]
            individual_predictions[model_name] = [str(confidence_score), predicted_label]
            predictions.append(pred)

        # Convert predictions to class labels
        predictions = np.array(predictions)
        predicted_classes = np.argmax(predictions, axis=-1)

        # Ensemble prediction by majority vote
        ensemble_preds, _ = mode(predicted_classes, axis=0)
        ensemble_label = labels[ensemble_preds[0]]
        individual_predictions['ensem']=ensemble_label

        return individual_predictions

    models = [model1, model2, model3]
    image_path = file_path
    predictions = ensemble_predictions(models, image_path)

    # Output the results
   

    print('Individual model predictions with confidence scores:')

    print(f"{predictions}")

    print(f'Ensemble prediction: {predictions['ensem']}')
    
    return predictions['ensem']

