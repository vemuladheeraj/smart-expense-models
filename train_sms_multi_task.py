import argparse
import os
import sys
import json
import re
import random
from typing import Tuple, List, Dict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
import sentencepiece as spm


def build_multi_task_model(vocab_size: int = 8000, sequence_length: int = 200, embedding_dim: int = 64) -> tf.keras.Model:
    """Build multi-task model for classification + entity extraction"""
    
    # Input layer for token IDs
    text_input = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32, name="input_ids")
    
    # Shared embedding and CNN layers
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim, name="embedding")(text_input)
    
    # CNN layers for feature extraction
    x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same', name="conv1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout1")(x)
    
    x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same', name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout2")(x)
    
    x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same', name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout3")(x)
    
    # Task 1: Binary Classification (transactional or not)
    classification_branch = tf.keras.layers.GlobalMaxPooling1D(name="classification_pool")(x)
    classification_branch = tf.keras.layers.Dense(128, activation='relu', name="class_dense1")(classification_branch)
    classification_branch = tf.keras.layers.BatchNormalization(name="class_bn1")(classification_branch)
    classification_branch = tf.keras.layers.Dropout(0.3, name="class_dropout1")(classification_branch)
    classification_branch = tf.keras.layers.Dense(64, activation='relu', name="class_dense2")(classification_branch)
    classification_branch = tf.keras.layers.BatchNormalization(name="class_bn2")(classification_branch)
    classification_branch = tf.keras.layers.Dropout(0.3, name="class_dropout2")(classification_branch)
    classification_output = tf.keras.layers.Dense(1, activation='sigmoid', name="classification")(classification_branch)
    
    # Task 2: Entity Extraction using CNN layers (sequence-aware)
    # Use 1D convolutions with different kernel sizes to capture entity patterns
    entity_branch = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same', name="entity_conv1")(x)
    entity_branch = tf.keras.layers.BatchNormalization(name="entity_bn1")(entity_branch)
    entity_branch = tf.keras.layers.Dropout(0.2, name="entity_dropout1")(entity_branch)
    
    entity_branch = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same', name="entity_conv2")(entity_branch)
    entity_branch = tf.keras.layers.BatchNormalization(name="entity_bn2")(entity_branch)
    entity_branch = tf.keras.layers.Dropout(0.2, name="entity_dropout2")(entity_branch)
    
    # Entity extraction heads using CNN features
    # Merchant/User name extraction (BIO tagging: B-MERCHANT, I-MERCHANT, O)
    merchant_output = tf.keras.layers.Dense(3, activation='softmax', name="merchant_ner")(entity_branch)  # 3 classes: B-MERCHANT, I-MERCHANT, O
    
    # Amount extraction (BIO tagging: B-AMOUNT, I-AMOUNT, O)
    amount_output = tf.keras.layers.Dense(3, activation='softmax', name="amount_ner")(entity_branch)  # 3 classes: B-AMOUNT, I-AMOUNT, O
    
    # Transaction type extraction (BIO tagging: B-TYPE, I-TYPE, O)
    type_output = tf.keras.layers.Dense(3, activation='softmax', name="type_ner")(entity_branch)  # 3 classes: B-TYPE, I-TYPE, O
    
    # Transaction direction (debit/credit) - using global features
    direction_branch = tf.keras.layers.GlobalMaxPooling1D(name="direction_pool")(x)
    direction_branch = tf.keras.layers.Dense(64, activation='relu', name="direction_dense1")(direction_branch)
    direction_branch = tf.keras.layers.BatchNormalization(name="direction_bn1")(direction_branch)
    direction_branch = tf.keras.layers.Dropout(0.3, name="direction_dropout1")(direction_branch)
    direction_branch = tf.keras.layers.Dense(32, activation='relu', name="direction_dense2")(direction_branch)
    direction_branch = tf.keras.layers.BatchNormalization(name="direction_bn2")(direction_branch)
    direction_branch = tf.keras.layers.Dropout(0.3, name="direction_dropout2")(direction_branch)
    direction_output = tf.keras.layers.Dense(3, activation='softmax', name="direction")(direction_branch)  # 3 classes: debit, credit, none
    
    model = tf.keras.Model(
        inputs=text_input, 
        outputs=[
            classification_output,      # Binary classification
            merchant_output,           # Merchant/user name NER
            amount_output,            # Amount NER
            type_output,              # Transaction type NER
            direction_output          # Debit/credit classification
        ],
        name="sms_multi_task_classifier"
    )
    return model


def create_training_labels(texts: List[str], labels: List[int]) -> Tuple[List[int], List[List[int]], List[List[int]], List[List[int]], List[int]]:
    """Create training labels for multi-task learning"""
    
    # Classification labels (already available)
    classification_labels = labels
    
    # Entity extraction labels (BIO tagging)
    merchant_labels = []
    amount_labels = []
    type_labels = []
    direction_labels = []
    
    for text, label in zip(texts, labels):
        if label == 0:  # Not transactional
            # All entities are "O" (outside)
            merchant_labels.append([0] * 200)  # 0 = O
            amount_labels.append([0] * 200)    # 0 = O
            type_labels.append([0] * 200)      # 0 = O
            direction_labels.append(2)          # 2 = none
        else:
            # Transactional - create entity labels
            merchant_seq = [0] * 200  # 0 = O
            amount_seq = [0] * 200    # 0 = O
            type_seq = [0] * 200      # 0 = O
            
            # Simple rule-based entity labeling
            words = text.lower().split()
            for i, word in enumerate(words):
                if i >= 200:
                    break
                    
                # Merchant detection (common merchant names)
                if any(merchant in word for merchant in ['amazon', 'flipkart', 'swiggy', 'zomato', 'uber', 'ola', 'paytm', 'phonepe']):
                    merchant_seq[i] = 1  # 1 = B-MERCHANT
                
                # Amount detection (Rs. pattern)
                if 'rs.' in word or any(char.isdigit() for char in word):
                    amount_seq[i] = 1  # 1 = B-AMOUNT
                
                # Transaction type detection
                if any(txn_type in word for txn_type in ['upi', 'neft', 'rtgs', 'imps', 'atm', 'pos']):
                    type_seq[i] = 1  # 1 = B-TYPE
            
            merchant_labels.append(merchant_seq)
            amount_labels.append(amount_seq)
            type_labels.append(type_seq)
            
            # Direction detection
            if any(word in text.lower() for word in ['debited', 'debit', 'paid', 'sent']):
                direction_labels.append(0)  # 0 = debit
            elif any(word in text.lower() for word in ['credited', 'credit', 'received', 'refund']):
                direction_labels.append(1)  # 1 = credit
            else:
                direction_labels.append(2)  # 2 = none
    
    return classification_labels, merchant_labels, amount_labels, type_labels, direction_labels


def main():
    parser = argparse.ArgumentParser(description="Train multi-task SMS classifier with entity extraction")
    parser.add_argument("--output_dir", default="artifacts_multi_task", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load existing trained model and tokenizer
    print("Loading existing model and tokenizer...")
    
    # Load tokenizer
    sp_model = spm.SentencePieceProcessor()
    sp_model.load("sms_tokenizer.model")
    
    # Load some sample data for demonstration
    sample_texts = [
        "Rs.5000 debited from A/c XXXX1234 for UPI transaction to Amazon on 15/12/2024",
        "Your OTP for Net Banking is 123456. Valid for 10 minutes.",
        "Get 50% off on Electronics at Amazon! Use code SAVE50. Limited time offer.",
        "Dear Customer, Rs.2500 credited to A/c XXXX5678 for Salary on 01/12/2024"
    ]
    
    sample_labels = [1, 0, 0, 1]  # 1 = transactional, 0 = not transactional
    
    # Create multi-task labels
    classification_labels, merchant_labels, amount_labels, type_labels, direction_labels = create_training_labels(
        sample_texts, sample_labels
    )
    
    # Preprocess texts
    tokenized_texts = []
    for text in sample_texts:
        tokens = sp_model.encode_as_ids(text)
        if len(tokens) > 200:
            tokens = tokens[:200]
        else:
            tokens = tokens + [0] * (200 - len(tokens))
        tokenized_texts.append(tokens)
    
    X = np.array(tokenized_texts)
    y_classification = np.array(classification_labels)
    y_merchant = np.array(merchant_labels)
    y_amount = np.array(amount_labels)
    y_type = np.array(type_labels)
    y_direction = np.array(direction_labels)
    
    # Build multi-task model
    model = build_multi_task_model(vocab_size=8000, sequence_length=200, embedding_dim=64)
    
    # Compile model with multi-task losses
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'classification': 'binary_crossentropy',
            'merchant_ner': 'sparse_categorical_crossentropy',
            'amount_ner': 'sparse_categorical_crossentropy', 
            'type_ner': 'sparse_categorical_crossentropy',
            'direction': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'classification': 1.0,
            'merchant_ner': 0.5,
            'amount_ner': 0.5,
            'type_ner': 0.5,
            'direction': 0.5
        },
        metrics={
            'classification': ['accuracy'],
            'merchant_ner': ['accuracy'],
            'amount_ner': ['accuracy'],
            'type_ner': ['accuracy'],
            'direction': ['accuracy']
        }
    )
    
    print("Model summary:")
    model.summary()
    
    # Train model
    print("Training multi-task model...")
    history = model.fit(
        X, {
            'classification': y_classification,
            'merchant_ner': y_merchant,
            'amount_ner': y_amount,
            'type_ner': y_type,
            'direction': y_direction
        },
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2
    )
    
    # Test inference
    print("\nTesting multi-task inference...")
    for text in sample_texts:
        tokens = sp_model.encode_as_ids(text)
        if len(tokens) > 200:
            tokens = tokens[:200]
        else:
            tokens = tokens + [0] * (200 - len(tokens))
        
        predictions = model.predict(np.array([tokens]), verbose=0)
        
        # Extract predictions for each task
        classification_pred = predictions[0][0][0]
        merchant_ner_pred = predictions[1][0]  # Shape: (200, 3)
        amount_ner_pred = predictions[2][0]   # Shape: (200, 3)
        type_ner_pred = predictions[3][0]     # Shape: (200, 3)
        direction_pred = predictions[4][0]    # Shape: (3,)
        
        # Convert predictions to labels
        classification_label = "transactional" if classification_pred > 0.5 else "not_transactional"
        direction_label = ["debit", "credit", "none"][np.argmax(direction_pred)]
        
        print(f"Text: {text[:50]}...")
        print(f"Classification: {classification_pred:.4f} -> {classification_label}")
        print(f"Direction: {direction_pred} -> {direction_label}")
        print(f"Merchant NER shape: {merchant_ner_pred.shape}")
        print(f"Amount NER shape: {amount_ner_pred.shape}")
        print(f"Type NER shape: {type_ner_pred.shape}")
        print()
    
    # Save model
    saved_model_dir = os.path.join(args.output_dir, "saved_model")
    model.export(saved_model_dir)
    
    # Convert to TFLite
    tflite_path = os.path.join(args.output_dir, "sms_multi_task.tflite")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # Ensure pure TFLite compatibility
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    
    # Additional flags for compatibility
    converter.experimental_enable_resource_variables = True
    converter._experimental_lower_tensor_list_ops = False
    
    try:
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite conversion successful: {tflite_path}")
    except Exception as e:
        print(f"TFLite conversion failed: {e}")
        print("Trying with Select TF ops as fallback...")
        
        # Fallback: Use Select TF ops if pure TFLite fails
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite conversion with Select TF ops successful: {tflite_path}")
        print("Note: This model requires Select TF ops and may not be pure TFLite")
    
    print(f"Multi-task model saved to: {args.output_dir}")
    print(f"- TFLite model: {tflite_path}")
    print(f"- SavedModel: {saved_model_dir}")


if __name__ == "__main__":
    main()
