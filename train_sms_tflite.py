import argparse
import os
import sys
import json
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_dataset(csv_path: str) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)
    expected_cols = {"adress", "body", "label"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

    # Clean and normalize
    df["body"] = df["body"].astype(str).fillna("")
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Map labels: transactional vs not_transactional
    label_map = {"transactional": 1, "not_transactional": 0}
    if not set(df["label"]).issubset(set(label_map.keys())):
        unknown = sorted(set(df["label"]) - set(label_map.keys()))
        raise ValueError(f"Unknown labels present: {unknown}. Expected only {list(label_map.keys())}")
    df["target"] = df["label"].map(label_map)
    return df["body"], df["target"]


def build_int_model(
    vocab_size: int = 20000,
    sequence_length: int = 200,
    embedding_dim: int = 64,
) -> tf.keras.Model:
    # Pure TFLite built-ins: int32 token ids -> Embedding -> Conv1D -> GlobalMaxPool -> Dense
    token_input = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32, name="tokens")
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim, name="embedding")(token_input)
    x = tf.keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="probability")(x)
    model = tf.keras.Model(token_input, output)
    return model


def export_label_map(export_dir: str, id_to_label: dict) -> None:
    path = os.path.join(export_dir, "labels.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(id_to_label, f, ensure_ascii=False, indent=2)


def export_vocabulary(export_dir: str, vocab: List[str]) -> None:
    path = os.path.join(export_dir, "vocab.txt")
    with open(path, "w", encoding="utf-8") as f:
        for tok in vocab:
            f.write(tok + "\n")


def convert_to_tflite(saved_model_dir: str, tflite_path: str) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    # Pure TFLite built-ins only
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TFLite SMS transactional classifier")
    parser.add_argument("--csv", type=str, default="neatsmsdata.csv", help="Path to training CSV")
    parser.add_argument("--output_dir", type=str, default="artifacts", help="Where to write models")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--vocab_size", type=int, default=20000, help="Max vocabulary size")
    parser.add_argument("--seq_len", type=int, default=200, help="Sequence length for vectorizer")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    texts, targets = load_dataset(args.csv)

    X_train, X_val, y_train, y_val = train_test_split(
        texts, targets, test_size=0.15, random_state=42, stratify=targets
    )

    print("Preparing vectorizer...")
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=args.vocab_size,
        output_mode="int",
        output_sequence_length=args.seq_len,
        standardize="lower_and_strip_punctuation",
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(X_train.values).batch(256))

    # Build pure-int input model
    print("Building model (pure TFLite)...")
    model = build_int_model(vocab_size=args.vocab_size, sequence_length=args.seq_len)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.BinaryAccuracy(name="acc")],
    )

    # Build datasets: vectorize to int tokens before feeding model
    def make_ds(texts_series: pd.Series, labels_series: pd.Series, shuffle: bool) -> tf.data.Dataset:
        text_ds = tf.data.Dataset.from_tensor_slices(texts_series.values)
        tok_ds = text_ds.batch(1024).map(vectorizer).unbatch()
        ds = tf.data.Dataset.zip((tok_ds, tf.data.Dataset.from_tensor_slices(labels_series.values.astype(np.float32))))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(texts_series), 10000), seed=42, reshuffle_each_iteration=True)
        return ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(X_train, y_train, shuffle=True)
    val_ds = make_ds(X_val, y_val, shuffle=False)

    # Compute class weights for imbalance
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    total = pos + neg
    class_weight = {0: total / (2.0 * neg), 1: total / (2.0 * pos)}
    print(f"Class weights: {class_weight}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=2, restore_best_weights=True),
    ]

    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    print("Evaluating...")
    eval_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)
    print(json.dumps({"val_metrics": eval_metrics}, indent=2))

    # Save SavedModel (int tokens input)
    saved_model_dir = os.path.join(args.output_dir, "saved_model")
    print(f"Saving SavedModel to {saved_model_dir} ...")
    model.export(saved_model_dir)

    # Save label map
    export_label_map(args.output_dir, {"0": "not_transactional", "1": "transactional"})
    # Save vocabulary for external tokenization
    export_vocabulary(args.output_dir, vectorizer.get_vocabulary())

    # Convert to TFLite
    tflite_path = os.path.join(args.output_dir, "sms_classifier.tflite")
    print(f"Converting to TFLite at {tflite_path} ...")
    convert_to_tflite(saved_model_dir, tflite_path)

    # Quick sanity inference
    examples = [
        "Rs. 500 credited to your account via UPI",
        "Flat 70% OFF on sale today!",
    ]
    # Vectorize examples externally
    ex_tokens = vectorizer(tf.constant(examples))
    preds = model.predict(ex_tokens, verbose=0).reshape(-1)
    print("Sample predictions:")
    for text, p in zip(examples, preds):
        print(f"  {p:.3f}  ->  {text}")

    print("Done.")


if __name__ == "__main__":
    # Improve determinism best-effort
    os.environ.setdefault("PYTHONHASHSEED", "42")
    tf.random.set_seed(42)
    np.random.seed(42)
    main()


