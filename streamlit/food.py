import tensorflow as tf
import numpy as np

# defining the classes 
class_names = ["apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake", "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla", "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"]
# Load the fine-tuned model checkpoint
model = tf.keras.models.load_model("C:/Users/Администратор/Desktop/food_app/tensorflow-deeplearning/streamlit/model_1.keras")
# Function to preprocess input image
def prepare_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.imagenet_utils.preprocess_input(img_array)
    return img_array

# Custom function to decode predictions for Food101
def decode_predictions_food(preds, top=1):
    top_indices = preds[0].argsort()[-top:][::-1]
    result = [(class_names[i], preds[0][i]) for i in top_indices]
    return result

# Function to make prediction
def predict_food(img_path):
    img = prepare_image(img_path)
    preds = model.predict(img)
    print('Predicted:', decode_predictions_food(preds, top=3)[0])

# Test the function
predict_food("pizza_1.jpg")
