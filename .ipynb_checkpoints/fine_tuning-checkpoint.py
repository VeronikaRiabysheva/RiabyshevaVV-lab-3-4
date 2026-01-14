from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

dataset = load_dataset("emotion")

small_train = dataset["train"].shuffle(seed=42).select(range(2000))
small_val = dataset["validation"].shuffle(seed=42).select(range(500))
small_test = dataset["test"].shuffle(seed=42).select(range(500))

model_name = "distilbert-base-uncased"  # Или "prajjwal1/bert-tiny" для ещё большего ускорения
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=128
    )

tokenized_datasets = {}
tokenized_datasets["train"] = small_train.map(tokenize_function, batched=True)
tokenized_datasets["validation"] = small_val.map(tokenize_function, batched=True)
tokenized_datasets["test"] = small_test.map(tokenize_function, batched=True)

for split in tokenized_datasets:
    tokenized_datasets[split] = tokenized_datasets[split].rename_column("label", "labels")
    tokenized_datasets[split].set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# DataCollator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Загрузка модели
num_labels = len(set(dataset["train"]["label"]))
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label={0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'},
    label2id={'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
)

# Метрики
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1_score": f1}

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_score",
    logging_dir="./logs",
    logging_steps=100,
    report_to="none",
    fp16=False,
    gradient_accumulation_steps=4,
    optim="adamw_torch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Обучение
print("Начало обучения...")
train_result = trainer.train()
trainer.save_model("./emotion-classifier")
tokenizer.save_pretrained("./emotion-classifier")
print("Обучение завершено!")
print(f"Результаты обучения: {train_result.metrics}")

# Оценка на test
test_results = trainer.evaluate(tokenized_datasets["test"])
print(f"Результаты на тестовых данных: {test_results}")

# Сохранение результатов
with open("test_results.txt", "w") as f:
    f.write(f"Accuracy: {test_results['eval_accuracy']:.4f}\n")
    f.write(f"F1 Score: {test_results['eval_f1_score']:.4f}\n")

# Функция предсказания
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=1).item()
    return model.config.id2label[predicted_class], predictions[0][predicted_class].item()

# Тест на примерах
test_texts = [
    "I am feeling absolutely wonderful today!",
    "This is making me so angry and frustrated",
    "I'm scared about what might happen tomorrow"
]
print("\nТестирование модели:")
for text in test_texts:
    emotion, confidence = predict_emotion(text)
    print(f"Текст: '{text}'")
    print(f"Предсказание: {emotion} (уверенность: {confidence:.3f})")
    print()