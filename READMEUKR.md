Українська версія
# Diplome – ML система для інженерії даних

## Опис
Програмна система забезпечення інженерії даних для машинного навчання: інтелектуальний аналіз даних та нейронні мережі.  
Проєкт реалізує повний цикл роботи з даними: **ETL → ML → API → UI → БД**.

## Основні можливості
- ETL‑модуль: очищення, трансформація, додавання ознак.
- Моделі ML: RandomForest та Neural Network.
- REST API (FastAPI): `/etl`, `/train`, `/predict`, `/predict_batch_file`, `/metrics`.
- Веб‑інтерфейс (HTML+JS) для взаємодії з системою.
- Метрики: accuracy, precision, recall, f1, confusion matrix.
- Docker‑інфраструктура для розгортання.

## Запуск (основний сценарій)
### Docker
```bash
docker-compose up --build

#API буде доступний за адресою: http://localhost:8000

##Зупинка проєкту
```bash
docker-compose down

## Перезапуск проєкту
```bash
docker-compose down
docker-compose up --build

## локальний запуск (для тестів)
```bash
pip install -r requirements.txt
uvicorn api:app --reload
