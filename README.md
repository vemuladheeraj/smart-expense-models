## SMS Transactional Classifier (TFLite)

This project trains a TensorFlow model to classify SMS messages as **transactional** or **not_transactional**, and exports a `.tflite` model that accepts raw text input.

### Dataset

- Input CSV: `neatsmsdata.csv` with columns: `adress`, `body`, `label`
- Labels expected: `transactional` or `not_transactional`

### Setup

1. Create a virtual environment (recommended)
2. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Train and Export

```bash
python train_sms_tflite.py --csv neatsmsdata.csv --output_dir artifacts --epochs 6 --batch_size 64
```

Artifacts produced in `artifacts/`:

- `saved_model/` TensorFlow SavedModel (includes TextVectorization)
- `sms_classifier.tflite` TFLite model (string input: a single SMS)
- `labels.json` mapping `{"0": "not_transactional", "1": "transactional"}`

### Inference with TFLite (Python)

```python
import json
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="artifacts/sms_classifier.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

def predict(text: str) -> float:
    # Input is scalar string tensor
    interpreter.resize_tensor_input(input_details["index"], [1], strict=True)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details["index"], np.array([text], dtype=np.object_))
    interpreter.invoke()
    prob = interpreter.get_tensor(output_details["index"])[0][0]
    return float(prob)

print(predict("Rs. 500 credited to your account via UPI"))
print(predict("70% OFF SALE today only!"))
```

### Notes

- The model embeds the tokenizer via `TextVectorization`, so TFLite input is raw text.
- Adjust `--vocab_size`, `--seq_len`, and `--epochs` for your dataset size and accuracy.

