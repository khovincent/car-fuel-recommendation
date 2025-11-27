import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
import matplotlib.cm as cm

# --- Configuration ---
CLASS_NAMES = ['Pertalite', 'Pertamax', 'Pertamax Turbo'] 
MODEL_PATH = 'modeleffi.keras'

# --- Page Setup ---
st.set_page_config(
    page_title="Car Gas Type Recommendation + Grad-CAM",
    page_icon="🚗",
    layout="centered"
)

# --- Load Model ---
@st.cache_resource
def load_prediction_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_prediction_model()

# --- Grad-CAM Helper Functions ---
def find_target_layer(model, layer_name='top_activation'):
    """
    Recursively search for the target layer (EfficientNet's 'top_activation')
    Handling cases where the model is a wrapper around the base model.
    """
    # 1. Check if layer exists directly in the model
    for layer in model.layers:
        if layer.name == layer_name:
            return model, layer.name
            
    # 2. Check if the model has a nested 'efficientnetb0' layer (common in transfer learning)
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or 'efficientnet' in layer.name.lower():
            # If found, return the nested model and the layer name
            return layer, layer_name
            
    # 3. Fallback: Return the last Conv2D layer found
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return model, layer.name
            
    return None, None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 1. Find the target layer (handling nested EfficientNet models)
    target_model_object, target_layer_name = find_target_layer(model, last_conv_layer_name)
    
    if target_model_object is None:
        st.error("Could not find the last convolutional layer. Grad-CAM failed.")
        return None

    # 2. Create a model that maps the input image to the activations of the last conv layer
    #    and the final predictions
    grad_model = tf.keras.models.Model(
        inputs=[target_model_object.inputs],
        outputs=[target_model_object.get_layer(target_layer_name).output, target_model_object.output]
    )

    # 3. Compute the gradient
    with tf.GradientTape() as tape:
        results = grad_model(img_array)
        
        # FIX: Handle Keras 3 / TF returning lists or single tensors differently
        if len(results) == 2:
            last_conv_layer_output = results[0]
            preds = results[1]
        else:
            # Fallback for unexpected return structures
            last_conv_layer_output = results[0]
            preds = results[-1]

        # EXTRA FIX: If preds is still a list (e.g. [Tensor]), unwrap it
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
            
        class_channel = preds[:, pred_index]

    # 4. Process Gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Check if grads is None (can happen if the layer is disconnected)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Handle list wrapping for the conv output as well
    if isinstance(last_conv_layer_output, (list, tuple)):
        last_conv_layer_output = last_conv_layer_output[0]

    last_conv_layer_output = last_conv_layer_output[0]
    
    # Matrix multiplication
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    
    # Resize the heatmap to match the original image size
    jet_heatmap = jet_heatmap.resize((img.size[0], img.size[1]))

    # Convert to array for overlaying
    jet_heatmap_array = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap_array * alpha + img_array
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # RETURN BOTH: The overlay AND the raw heatmap image
    return superimposed_img, jet_heatmap

# --- Preprocessing & Prediction Function ---
def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_reshape = img_array[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction, img_reshape

# --- User Interface ---
st.title("🚗 Car Gas Type Predictor")
st.markdown("Upload an image of a car to predict its fuel type and see **Grad-CAM heatmaps**.")

file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])

if file is None:
    st.text("Waiting for upload...")
else:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', width="stretch")
    
    if model is not None:
        if st.button("Predict & Analyze"):
            with st.spinner('Analyzing image...'):
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # Predict
                prediction, img_reshape = import_and_predict(image, model)
                
                score = tf.nn.softmax(prediction[0])
                confidence = np.max(score) * 100
                class_index = np.argmax(score)
                predicted_class = CLASS_NAMES[class_index]

                st.success(f"Prediction: **{predicted_class}**")
                st.info(f"Confidence: **{confidence:.2f}%**")

                # --- Grad-CAM Logic ---
                st.markdown("### 🔍 AI Explanation (Grad-CAM)")
                st.write("The red/yellow areas show where the model looked to make this decision.")

                heatmap = make_gradcam_heatmap(img_reshape, model, 'top_activation', pred_index=class_index)
                
                if heatmap is not None:
                    # Get BOTH images from the updated function
                    overlay_img, raw_heatmap_img = display_gradcam(image, heatmap)
                    
                    # Create two columns to display them side-by-side
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(overlay_img, caption=f"Overlay", width="stretch")
                    
                    with col2:
                        st.image(raw_heatmap_img, caption=f"Raw Heatmap", width="stretch")

                else:
                    st.warning("Could not generate heatmap. (Layer not found)")