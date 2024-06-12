from transformers import TFAutoModel, AutoTokenizer
from datasets import load_dataset
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling1D, Masking
import numpy as np
from sklearn.preprocessing import normalize

# Cosine Similarity function
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)

    # Clip the similarity value to be within [-1, 1]
    similarity = np.clip(similarity, -1.0, 1.0)

    return similarity

# Obtain BERT model
bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")

# Initialize BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load dataset (e.g., gender hate speech)
dataset = load_dataset("ctoraman/gender-hate-speech")

# Limit the dataset to the first 200 examples for both train and test splits
train_split = dataset["train"].shuffle(seed=0)[:2000]
test_split = dataset["test"].shuffle(seed=0)[:2000]

# Combine the modified splits into a new dataset dictionary
dataset = {"train": train_split, "test": test_split}

# Tokenize the training data with dynamic maxlen
text_column_name = "Text"
label_column_name = "Label"

# Shuffle the training dataset
train_dataset = dataset["train"]

tokenized_train = tokenizer(
    train_dataset[text_column_name],
    truncation=True,
    padding=True,
    return_tensors="tf"
)
train_labels = to_categorical(train_dataset[label_column_name], num_classes=3)

# Build a fine-tuning model with Masking layer
token_ids = Input(shape=(None,), dtype=tf.int32, name="token_ids")
attention_masks = Input(shape=(None,), dtype=tf.int32, name="attention_masks")
masking_layer = Masking(mask_value=0)(token_ids)  # Add Masking layer
bert_output = bert_model(masking_layer, attention_mask=attention_masks)
pooled_output = GlobalAveragePooling1D()(bert_output.last_hidden_state)
output = Dense(3, activation="softmax")(pooled_output)
model = Model(inputs=[token_ids, attention_masks], outputs=output)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(
    [tokenized_train["input_ids"], tokenized_train["attention_mask"]],
    train_labels,
    batch_size=25,
    epochs=3,
    validation_split=0.2
)

# Tokenize the test data with the same dynamic maxlen
test_dataset = dataset["test"]
tokenized_test = tokenizer(
    test_dataset[text_column_name],
    truncation=True,
    padding=True,
    return_tensors="tf"
)
test_labels = to_categorical(test_dataset[label_column_name], num_classes=3)

# Evaluate the model on the test data
score = model.evaluate(
    [tokenized_test["input_ids"], tokenized_test["attention_mask"]],
    test_labels,
    verbose=0
)
print("Accuracy on test data:", score[1])
print()
# Examples for Cosine Similarity Demonstration
example_pairs = [
    {"text1": "he kangaroo is hopping roo-jumps.", "text2": "The turtle is sunbathing."},
    {"text1": "The penguin is sliding ice-slips.", "text2": "The bear is hibernating."},
    {"text1": "The giraffe is stretching neckyreach.", "text2": "The monkey is swinging."},
    {"text1": "The dolphin is leaping sea-flips.", "text2": "The whale is singing."},
    {"text1": "The owl is hooting night-tunes.", "text2": "The rabbit is nibbling on grass."}
]

# Calculate and print cosine similarity scores for the example pairs
for example_pair in example_pairs:
    text1 = example_pair["text1"]
    text2 = example_pair["text2"]

    # Tokenize and obtain embeddings for text1
    input_ids_1 = tokenizer(text1, return_tensors='tf')['input_ids']
    attention_mask_1 = tokenizer(text1, return_tensors='tf')['attention_mask']
    output_1 = model.predict([input_ids_1, attention_mask_1])

    # Tokenize and obtain embeddings for text2
    input_ids_2 = tokenizer(text2, return_tensors='tf')['input_ids']
    attention_mask_2 = tokenizer(text2, return_tensors='tf')['attention_mask']
    output_2 = model.predict([input_ids_2, attention_mask_2])

    # Flatten the embeddings
    output_1_flat = output_1.flatten()
    output_2_flat = output_2.flatten()

    # Normalize embeddings
    output_1_norm = normalize(output_1_flat.reshape(1, -1))[0]
    output_2_norm = normalize(output_2_flat.reshape(1, -1))[0]

    # Calculate cosine similarity using the provided function
    cosine_similarity_value = cosine_similarity(output_1_flat, output_2_flat)

    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Cosine Similarity Score: {cosine_similarity_value}\n")

# Predictions on test data
predictions = model.predict([tokenized_test["input_ids"], tokenized_test["attention_mask"]])

print(predictions)

print(predictions.shape)
print(predictions[0])
print(predictions[1])

# Analysis of correct and incorrect predictions
correctly_predicted_indices = []
incorrectly_predicted_indices = []
correct_predictions = 0
incorrect_predictions = 0

# Iterate through the test dataset and make predictions
for i in range(len(predictions)):
    predicted_label = np.argmax(predictions[i])
    true_label = np.argmax(test_labels[i])

    if predicted_label == true_label:
        correctly_predicted_indices.append(i)
        correct_predictions += 1
    else:
        incorrectly_predicted_indices.append(i)
        incorrect_predictions += 1

    # Stop when we have at least 10 correct and 10 incorrect predictions
    if correct_predictions >= 10 and incorrect_predictions >= 10:
        break

# Print observations for correctly predicted examples
print("Observations for Correct Predictions:")
for idx in correctly_predicted_indices[:10]:
    print(f"Example {idx}:")
    print("Predicted:", np.argmax(predictions[idx]))
    print("True Label:", np.argmax(test_labels[idx]))
    print("Text:", test_dataset[text_column_name][idx])
    print()

# Print observations for incorrectly predicted examples
print("Observations for Incorrect Predictions:")
for idx in incorrectly_predicted_indices[:10]:
    print(f"Example {idx}:")
    print("Predicted:", np.argmax(predictions[idx]))
    print("True Label:", np.argmax(test_labels[idx]))
    print("Text:", test_dataset[text_column_name][idx])
    print()
