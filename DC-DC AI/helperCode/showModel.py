from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.utils import plot_model

# Load the trained model
model = load_model('trained_model.h5', custom_objects={'mse': MeanSquaredError()})

# Generate a graphical representation of the model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

print("Model plot saved as 'model_plot.png'")
