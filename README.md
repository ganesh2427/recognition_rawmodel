# Siamese Network for Face Verification

This project implements a Siamese Neural Network for face verification using TensorFlow and Keras. It includes a basic CNN-based embedding model and training on anchor-positive-negative image triplets. The network learns to distinguish between similar and dissimilar faces based on L1 distance in the embedding space.

## 📁 Project Structure

├── data/ │ ├── anchor/ │ ├── positive/ │ └── negative/ ├── efficientnet.ipynb # EfficientNet-based model (optional alternate architecture) ├── normal.ipynb # Non-EfficientNet version for testing/evaluation ├── face.py # Main training and model definition script ├── training_checkpoints/ # Stores model checkpoints └── siamesemodelv2.keras # Final trained model

markdown
Copy
Edit

## 📦 Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

Install the dependencies with:

```bash
pip install tensorflow opencv-python numpy matplotlib
🧠 Model Architecture
Embedding Network
A custom CNN-based embedding extractor with:

Conv2D and MaxPooling layers:

64 filters (10x10)

128 filters (7x7)

128 filters (4x4)

256 filters (4x4)

Flatten + Dense(4096, activation='sigmoid')

Siamese Network
Inputs: image pairs (anchor, positive/negative)

Embeddings: extracted using the embedding model

Distance: calculated using L1 (absolute) difference

Output: single neuron with sigmoid activation for similarity classification

🏋️ Training
Run the training process using:

bash
Copy
Edit
python face.py
Uses datasets from data/anchor, data/positive, data/negative

Preprocesses images to 100x100 and normalizes

Batches data and shuffles with a buffer of 10,000

Trains for 10 epochs (can be modified)

Saves model to siamesemodelv2.keras

Checkpoints saved every 10 epochs in training_checkpoints/

💾 Model Saving & Loading
To save the trained model:

python
Copy
Edit
siamese_model.save('siamesemodelv2.keras')
To reload the model:

python
Copy
Edit
siamese_model = tf.keras.models.load_model(
    'siamesemodelv2.keras',
    custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy}
)
📊 Evaluation
During training:

Prints loss, precision, and recall after each epoch

Shows progress bar for each batch

Evaluates using both positive and negative pairs

You can use normal.ipynb to visualize predictions or test new images with the trained model.

📸 Data Preparation
Prepare your dataset with the following structure:

pgsql
Copy
Edit
data/
├── anchor/
│   └── *.jpg (reference images)
├── positive/
│   └── *.jpg (same identity as anchor)
└── negative/
    └── *.jpg (different identity)
All images should be resized to 100x100

Ensure variety in poses, lighting, and backgrounds for better generalization

✅ Use Cases
Face verification systems

One-shot learning for identity matching

Authentication in access control systems

🧪 Future Improvements
Replace the CNN backbone with EfficientNet for better feature extraction

Use ArcFace or Triplet Loss for more robust training

Improve data preprocessing with facial landmark alignment

Add GUI for real-time verification

css
Copy
Edit

Let me know if you also want a `requirements.txt`, GitHub setup instructions, or a badge-style header!







