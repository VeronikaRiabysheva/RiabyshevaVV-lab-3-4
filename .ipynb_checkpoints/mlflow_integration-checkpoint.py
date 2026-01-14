import mlflow
import mlflow.transformers
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
import os

# Настройка MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Emotion-Classification-FineTuning")

with mlflow.start_run():
    # Загрузка датасета и уменьшение для ускорения
    dataset = load_dataset("emotion")
    small_train = dataset["train"].shuffle(seed=42).select(range(2000))
    small_val = dataset["validation"].shuffle(seed=42).select(range(500))
    small_test = dataset["test"].shuffle(seed=42).select(range(500))

    # Загрузка токенизатора
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Токенизация
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

    # Параметры модели и обучения
    model_params = {
        "model_name": model_name,
        "num_labels": 6,
        "learning_rate": 2e-5,
        "batch_size": 32,
        "num_epochs": 1,
        "weight_decay": 0.01
    }
    mlflow.log_params(model_params)

    # Загрузка модели
    model = AutoModelForSequenceClassification.from_pretrained(
        model_params["model_name"],
        num_labels=model_params["num_labels"],
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

    # Гиперпараметры с оптимизациями
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=model_params["learning_rate"],
        per_device_train_batch_size=model_params["batch_size"],
        per_device_eval_batch_size=model_params["batch_size"],
        num_train_epochs=model_params["num_epochs"],
        weight_decay=model_params["weight_decay"],
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

    # DataCollator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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

    # Обучение с логированием метрик
    print("Начало обучения с трекингом в MLflow...")
    train_result = trainer.train()

    # Логирование метрик обучения
    mlflow.log_metrics({
        "train_loss": train_result.metrics.get("train_loss", 0),
        "eval_loss": train_result.metrics.get("eval_loss", 0),
        "eval_accuracy": train_result.metrics.get("eval_accuracy", 0),
        "eval_f1_score": train_result.metrics.get("eval_f1_score", 0)
    })

    # Оценка на тестовых данных
    test_results = trainer.evaluate(tokenized_datasets["test"])
    mlflow.log_metrics({
        "test_accuracy": test_results["eval_accuracy"],
        "test_f1_score": test_results["eval_f1_score"]
    })

    # Сохранение и логирование модели
    model_path = "./emotion-classifier-mlflow"
    trainer.save_model(model_path)
    mlflow.transformers.log_model(
        transformers_model={
            "model": model,
            "tokenizer": tokenizer
        },
        artifact_path="emotion-classifier",
        registered_model_name="distilbert-emotion-classifier"
    )

    # Логирование дополнительных артефактов
    with open("training_summary.txt", "w") as f:
        f.write(f"Training completed successfully!\n")
        f.write(f"Final training loss: {train_result.metrics.get('train_loss', 0):.4f}\n")
        f.write(f"Validation accuracy: {train_result.metrics.get('eval_accuracy', 0):.4f}\n")
        f.write(f"Test accuracy: {test_results['eval_accuracy']:.4f}\n")
    mlflow.log_artifact("training_summary.txt")
    print("Эксперимент успешно завершен и записан в MLflow!")