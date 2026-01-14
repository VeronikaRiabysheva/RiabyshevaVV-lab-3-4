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
from accelerate import Accelerator  # Для unwrap_model

# Настройка MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Emotion-Classification-FineTuning")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1_score": f1}

def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=128  # Исправлено с 1 на 128
    )

# Начало эксперимента MLflow
with mlflow.start_run():
    # Загрузка и подготовка данных
    dataset = load_dataset("emotion")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Параметры модели и обучения
    model_params = {
        "model_name": "distilbert-base-uncased",
        "num_labels": 6,
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 3,
        "weight_decay": 0.01
    }
    # Логирование параметров
    mlflow.log_params(model_params)

    # Загрузка модели
    model = AutoModelForSequenceClassification.from_pretrained(
        model_params["model_name"],
        num_labels=model_params["num_labels"],
        id2label={0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'},
        label2id={'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
    )

    # Настройка обучения
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=model_params["learning_rate"],
        per_device_train_batch_size=model_params["batch_size"],
        per_device_eval_batch_size=model_params["batch_size"],
        num_train_epochs=model_params["num_epochs"],
        weight_decay=model_params["weight_decay"],
        eval_strategy="epoch",  # Исправлено с evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        logging_dir="./logs",
        logging_steps=100,
        report_to="none",
        fp16=True  # Включили для GPU ускорения; если ошибка — поставь False
    )

    tokenizer = AutoTokenizer.from_pretrained(model_params["model_name"])  # Выносим tokenizer наружу для reuse
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Создание тренера
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
        "train_loss": train_result.training_loss,  # Исправлено: training_loss вместо metrics["train_loss"]
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

    # Разворачиваем модель для корректного логирования (фикс PicklingError)
    unwrapped_model = trainer.accelerator.unwrap_model(model)

    # Сохранение и логирование модели
    model_path = "./emotion-classifier-mlflow"
    trainer.save_model(model_path)

    # Логирование модели в MLflow
    mlflow.transformers.log_model(
        transformers_model={
            "model": unwrapped_model,  # Используем unwrapped_model
            "tokenizer": tokenizer
        },
        artifact_path="emotion-classifier",
        registered_model_name="distilbert-emotion-classifier"
    )

    # Логирование дополнительных артефактов
    with open("training_summary.txt", "w") as f:
        f.write(f"Training completed successfully!\n")
        f.write(f"Final training loss: {train_result.training_loss:.4f}\n")
        f.write(f"Validation accuracy: {train_result.metrics.get('eval_accuracy', 0):.4f}\n")
        f.write(f"Test accuracy: {test_results['eval_accuracy']:.4f}\n")
    mlflow.log_artifact("training_summary.txt")

    print("Эксперимент успешно завершен и записан в MLflow!")


def train_model(learning_rate=2e-5):
    # Копируем логику из основного скрипта, но с переопределением learning_rate
    model_params['learning_rate'] = learning_rate  # Переопределяем LR

    # Загрузка модели (повтор)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_params["model_name"],
        num_labels=model_params["num_labels"],
        id2label={0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'},
        label2id={'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
    )

    # TrainingArgs с новым LR
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
        fp16=True  # Или False, если без GPU
    )

    tokenizer = AutoTokenizer.from_pretrained(model_params["model_name"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()

    # Eval на валидации (для метрик)
    eval_results = trainer.evaluate(tokenized_datasets["validation"])

    # Возвращаем метрики
    return {
        "eval_loss": eval_results.get("eval_loss", 0),
        "eval_accuracy": eval_results.get("eval_accuracy", 0),
        "eval_f1_score": eval_results.get("eval_f1_score", 0)
    }