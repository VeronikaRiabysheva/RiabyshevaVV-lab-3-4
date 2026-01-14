from datasets import load_dataset
from huggingface_hub import list_models, list_datasets
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import torch
import numpy as np

print("Доступные датасеты для текстовой классификации:")
datasets = list_datasets(filter="task_categories:text-classification")
for dataset in datasets:
    print(f"- {dataset.id}")

print("\nЗагрузка датасета emotion...")
dataset = load_dataset("emotion")

print(f"\nСтруктура датасета: {dataset}")
print(f"\nПримеры из train split:")
train_df = pd.DataFrame(dataset['train'][:5])
print(train_df)

print("\nРаспределение классов в тренировочных данных:")
label_counts = pd.Series(dataset['train']['label']).value_counts()
print(label_counts)

labels = ['sadness (0)', 'joy (1)', 'love (2)', 'anger (3)', 'fear (4)', 'surprise (5)']
plt.bar(labels, label_counts.sort_index())
plt.title('Распределение классов в train')
plt.ylabel('Количество')
plt.xticks(rotation=45)
plt.savefig('class_distribution.png')
print("\nГрафик распределения сохранён как class_distribution.png")

# Дополнительно: Статистика по длине текстов (по словам)
text_lengths = [len(text.split()) for text in dataset['train']['text']]
print("\nСтатистика по длине текстов в train (в словах):")
print(f"Средняя: {np.mean(text_lengths):.2f}")
print(f"Медиана: {np.median(text_lengths):.2f}")
print(f"Минимальная: {np.min(text_lengths)}")
print(f"Максимальная: {np.max(text_lengths)}")

print("\n\nДоступные модели для текстовой классификации:")
models = list_models(
    filter="task:text-classification",
    sort="downloads",
    direction=-1,
    limit=5
)
for model in models:
    print(f"\nМодель: {model.id}")
    print(f"Загрузок: {model.downloads}")
    print(f"Тэги: {model.tags}")
    if model.pipeline_tag:
        print(f"Тип задачи: {model.pipeline_tag}")

model_name = "distilbert-base-uncased"
print(f"\nЗагрузка модели {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=6
)
print("Модель и токенизатор успешно загружены!")
print(f"Размер словаря: {tokenizer.vocab_size}")
print(f"Архитектура модели: {model.__class__.__name__}")

test_text = "I am feeling very happy today!"
print(f"\nТекст для теста: {test_text}")
tokens = tokenizer(test_text, return_tensors="pt")
print(f"Токены: {tokens}")
print(f"Декодированные токены: {tokenizer.decode(tokens['input_ids'][0])}")

print("\nТестирование модели на 5 примерах из test:")
examples = dataset['test'][:5]
for i, (text, true_label) in enumerate(zip(examples['text'], examples['label'])):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred_label = torch.argmax(outputs.logits, dim=1).item()
    print(f"Пример {i+1}: Текст: '{text}'")
    print(f"True label: {true_label}")
    print(f"Pred label: {pred_label}")