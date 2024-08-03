import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification

# File paths
df1_path = 'archive/SQLiV3.csv'
df2_path = 'archive/sqli.csv'
df3_path = 'archive/sqliv2.csv'

# Load data
df1 = pd.read_csv(df1_path, encoding='utf-8')
df2 = pd.read_csv(df2_path, encoding='utf-16')
df3 = pd.read_csv(df3_path, encoding='utf-16')

# Select relevant columns
df1 = df1[["Sentence", "Label"]]
df2 = df2[["Sentence", "Label"]]
df3 = df3[["Sentence", "Label"]]

# Concatenate data
df = pd.concat([df1, df2, df3])

# Clean data
df.dropna(inplace=True)
df = df[(df['Label'] == "0") | (df['Label'] == "1")]
df = df.drop_duplicates(subset='Sentence')
df["Label"] = pd.to_numeric(df["Label"])

# Rename columns
df.rename(columns={'Sentence': 'X', 'Label': 'y'}, inplace=True)

# Split data
slice_index_1 = int(0.8 * len(df))
slice_index_2 = int(0.9 * len(df))
train_df = df.iloc[:slice_index_1, :]
val_df = df.iloc[slice_index_1:slice_index_2, :]
test_df = df.iloc[slice_index_2:, :]

X_train = train_df
y_train = X_train.pop('y').to_frame()

X_val = val_df
y_val = X_val.pop('y').to_frame()

X_test = test_df
y_test = X_test.pop('y').to_frame()

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenize and encode data
def tokenize_and_encode(df):
    encodings = tokenizer(df['X'].tolist(), padding=True, truncation=True, max_length=128, return_tensors='tf')
    return encodings

train_encodings = tokenize_and_encode(train_df)
val_encodings = tokenize_and_encode(val_df)
test_encodings = tokenize_and_encode(test_df)

# Convert DataFrame labels to TensorFlow tensors
train_labels = tf.convert_to_tensor(y_train['y'].tolist())
val_labels = tf.convert_to_tensor(y_val['y'].tolist())
test_labels = tf.convert_to_tensor(y_test['y'].tolist())

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask']
}, train_labels)).batch(8)

val_dataset = tf.data.Dataset.from_tensor_slices(({
    'input_ids': val_encodings['input_ids'],
    'attention_mask': val_encodings['attention_mask']
}, val_labels)).batch(8)

test_dataset = tf.data.Dataset.from_tensor_slices(({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask']
}, test_labels)).batch(8)

# Compile and train the model
optimizer = Adam(learning_rate=3e-5)
bert_model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = bert_model.fit(train_dataset, validation_data=val_dataset, epochs=3)

# Save the model
