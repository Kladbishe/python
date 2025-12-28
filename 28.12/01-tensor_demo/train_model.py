"""
Скрипт для обучения улучшенной модели распознавания собак
Использует Transfer Learning с MobileNetV2
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

# Настройки
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001

# Путь к данным (создайте эту структуру папок)
DATA_DIR = 'dataset/train'

print("=" * 60)
print("🐕 ОБУЧЕНИЕ МОДЕЛИ РАСПОЗНАВАНИЯ СОБАК")
print("=" * 60)

# Проверка наличия данных
if not os.path.exists(DATA_DIR):
    print(f"\n❌ ОШИБКА: Папка {DATA_DIR} не найдена!")
    print("\n📁 Создайте следующую структуру папок:")
    print("""
    dataset/
    └── train/
        ├── dog/
        │   ├── dog1.jpg
        │   ├── dog2.jpg
        │   └── ... (минимум 200 фото)
        └── no_dog/
            ├── cat1.jpg
            ├── person1.jpg
            └── ... (минимум 200 фото)
    """)
    exit(1)

# 1. Подготовка данных с аугментацией
print("\n📊 Подготовка данных...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,           # Поворот до ±20°
    width_shift_range=0.2,       # Сдвиг по горизонтали
    height_shift_range=0.2,      # Сдвиг по вертикали
    horizontal_flip=True,        # Горизонтальное отражение
    zoom_range=0.2,              # Зум
    brightness_range=[0.8, 1.2], # Изменение яркости
    fill_mode='nearest',
    validation_split=0.2         # 20% для валидации
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

print(f"✅ Найдено {train_generator.samples} обучающих изображений")
print(f"✅ Найдено {validation_generator.samples} валидационных изображений")
print(f"✅ Классы: {train_generator.class_indices}")

# 2. Создание модели
print("\n🏗️  Создание модели...")

# Загрузка предобученной модели MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Заморозка базовой модели (Transfer Learning)
base_model.trainable = False

# Добавление собственных слоёв
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print("✅ Модель создана")
print(f"📊 Всего параметров: {model.count_params():,}")

# 3. Компиляция модели
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Настройка callbacks
callbacks = [
    # Ранняя остановка при переобучении
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),

    # Уменьшение скорости обучения при застое
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),

    # Сохранение лучшей модели
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# 5. Обучение
print("\n🚀 Начало обучения...")
print(f"⏱️  Максимум эпох: {EPOCHS}")
print(f"📦 Размер батча: {BATCH_SIZE}")
print(f"📈 Learning rate: {LEARNING_RATE}")
print("-" * 60)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# 6. Сохранение финальной модели
print("\n💾 Сохранение модели...")
model.save('improved_dog_detector.h5')

# Создание файла меток
with open('improved_labels.txt', 'w') as f:
    f.write("0 no dog\n")
    f.write("1 Dog\n")

print("✅ Модель сохранена: improved_dog_detector.h5")
print("✅ Метки сохранены: improved_labels.txt")

# 7. Оценка результатов
print("\n" + "=" * 60)
print("📊 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
print("=" * 60)

final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"Финальная точность (обучение): {final_train_acc:.4f}")
print(f"Финальная точность (валидация): {final_val_acc:.4f}")
print(f"Финальная потеря (обучение): {final_train_loss:.4f}")
print(f"Финальная потеря (валидация): {final_val_loss:.4f}")

# 8. Визуализация (опционально)
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))

    # График точности
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # График потерь
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("\n📈 График сохранён: training_history.png")

except ImportError:
    print("\n⚠️  matplotlib не установлен, график не создан")

print("\n" + "=" * 60)
print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print("=" * 60)
print("\n📝 Следующие шаги:")
print("1. Замените 'keras_model.h5' на 'improved_dog_detector.h5'")
print("2. Замените 'labels.txt' на 'improved_labels.txt'")
print("3. Запустите веб-приложение: python web_app.py")
print("\n🎯 Удачи!")
