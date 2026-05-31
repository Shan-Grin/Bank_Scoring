import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
from flask import Flask, request, render_template
import os

# ==================== 1. ЗАГРУЗКА ДАННЫХ ====================
print("🔄 Загрузка данных...")

if os.path.exists('credit_risk_dataset.csv'):
    df = pd.read_csv('credit_risk_dataset.csv')
    print(f"✅ Загружено: {df.shape[0]} записей")

    if 'loan_status' in df.columns:
        if df['loan_status'].dtype == 'object':
            df['target'] = df['loan_status'].map({'fully paid': 0, 'charged off': 1})
        else:
            df['target'] = df['loan_status']
        print(f"✅ Целевая переменная создана")
    else:
        print("❌ Столбец 'loan_status' не найден")
        exit()
else:
    print("❌ Файл не найден")
    exit()

# ==================== 2. ОЧИСТКА ДАННЫХ ====================
print("🔄 Очистка данных...")

df = df.dropna(subset=['target'])
print(f"   После удаления NaN в target: {df.shape[0]} записей")

df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
df['loan_int_rate'] = df['loan_int_rate'].fillna(df.groupby('loan_grade')['loan_int_rate'].transform('median'))
df = df.dropna()

print(f"   После заполнения пропусков: {df.shape[0]} записей")

if df.shape[0] == 0:
    print("❌ Нет данных для обучения!")
    exit()

# Сохраняем исходные данные для страницы информации
original_df = df.copy()

# ==================== 3. ПРЕДОБРАБОТКА ====================
print("🔄 Предобработка...")

df['Geography_Russia'] = 1
df['loan_to_income'] = df['loan_amnt'] / (df['person_income'] + 1)
df['high_risk_grade'] = df['loan_grade'].isin(['E', 'F', 'G']).astype(int)
df['high_dti'] = (df['loan_percent_income'] > 0.4).astype(int)

grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df['loan_grade_enc'] = df['loan_grade'].map(grade_map)
df['default_on_file'] = df['cb_person_default_on_file'].map({'N': 0, 'Y': 1})

# One-Hot кодирование
home_dummies = pd.get_dummies(df['person_home_ownership'], prefix='home')
home_dummies = home_dummies[[c for c in ['home_RENT', 'home_OWN', 'home_MORTGAGE'] if c in home_dummies.columns]]
df = pd.concat([df, home_dummies], axis=1)

intent_dummies = pd.get_dummies(df['loan_intent'], prefix='intent')
df = pd.concat([df, intent_dummies], axis=1)

drop_cols = ['loan_status', 'loan_grade', 'cb_person_default_on_file', 'person_home_ownership', 'loan_intent']
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

X = df.drop('target', axis=1)
y = df['target']

print(f"📊 Размер: {X.shape[0]} записей, {X.shape[1]} признаков")
print(f"🎯 Доля оттока (дефолтов): {y.mean() * 100:.1f}%")

# Масштабирование
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
exclude = ['loan_grade_enc', 'default_on_file', 'high_risk_grade', 'high_dti', 'Geography_Russia',
           'home_RENT', 'home_OWN', 'home_MORTGAGE']
for col in intent_dummies.columns:
    exclude.append(col)
numeric_cols = [c for c in numeric_cols if c not in exclude]

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==================== 4. ОБУЧЕНИЕ ====================
print("🔄 Обучение модели...")
model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
recall = classification_report(y_test, y_pred, output_dict=True)['1']['recall']

print(f"\n✅ МОДЕЛЬ ОБУЧЕНА!")
print(f"   📊 Accuracy:  {accuracy:.3f} ({accuracy * 100:.1f}%)")
print(f"   📊 ROC-AUC:   {roc_auc:.3f}")
print(f"   📊 Recall (отток): {recall:.3f}")

joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Модель сохранена")

# ==================== 5. ПОДГОТОВКА ДАННЫХ ДЛЯ СТРАНИЦЫ ИНФОРМАЦИИ ====================
# Словарь для перевода названий признаков на русский язык
feature_names_ru = {
    # Исходные признаки
    'person_age': 'Возраст (лет)',
    'person_income': 'Годовой доход (₽)',
    'person_emp_length': 'Стаж работы (лет)',
    'loan_amnt': 'Сумма кредита (₽)',
    'loan_int_rate': 'Процентная ставка (%)',
    'loan_percent_income': 'Нагрузка (кредит/доход)',
    'cb_person_cred_hist_length': 'Кредитная история (лет)',
    'cb_person_default_on_file': 'Был дефолт ранее',
    'loan_grade': 'Кредитный рейтинг',
    'loan_intent': 'Цель кредита',
    'person_home_ownership': 'Тип жилья',
    # Созданные признаки
    'Geography_Russia': 'Россия (фиксация)',
    'loan_to_income': 'Отношение кредита к доходу',
    'high_risk_grade': 'Высокий риск (рейтинг E-G)',
    'high_dti': 'Высокая долговая нагрузка (>40%)',
    'loan_grade_enc': 'Кредитный рейтинг (число 1-7)',
    'default_on_file': 'Предыдущий дефолт (0/1)',
    # One-Hot признаки (тип жилья)
    'home_RENT': 'Тип жилья: Аренда',
    'home_OWN': 'Тип жилья: Собственность',
    'home_MORTGAGE': 'Тип жилья: Ипотека',
    # One-Hot признаки (цель кредита)
    'intent_EDUCATION': 'Цель: Образование',
    'intent_HOMEIMPROVEMENT': 'Цель: Ремонт',
    'intent_MEDICAL': 'Цель: Медицина',
    'intent_PERSONAL': 'Цель: Личные нужды',
    'intent_VENTURE': 'Цель: Бизнес',
    'intent_DEBTCONSOLIDATION': 'Цель: Объединение долгов',
}

# Получаем русские названия для всех признаков
feature_names_ru_list = [feature_names_ru.get(f, f) for f in feature_names]

# Статистика по исходным данным
data_stats = {
    'total_records': len(original_df),
    'features_count': len(feature_names),
    'churn_rate': float(original_df['target'].mean() * 100),
    'feature_names': feature_names,
    'feature_names_ru': feature_names_ru_list,
    'numeric_cols': numeric_cols,
    'categorical_cols': ['Тип жилья', 'Цель кредита', 'Кредитный рейтинг', 'Был дефолт ранее'],
    'sample_data': original_df.head(10).to_dict('records'),
    'model_params': {
        'n_estimators': 100,
        'max_depth': 12,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    },
    'model_metrics': {
        'accuracy': round(accuracy, 3),
        'roc_auc': round(roc_auc, 3),
        'recall': round(recall, 3)
    }
}

# ==================== 6. FLASK ====================
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predict():
    probability = None
    if request.method == 'POST':
        input_data = {
            'person_age': float(request.form['person_age']),
            'person_income': float(request.form['person_income']),
            'person_emp_length': float(request.form['person_emp_length']),
            'loan_amnt': float(request.form['loan_amnt']),
            'loan_int_rate': float(request.form['loan_int_rate']),
            'loan_percent_income': float(request.form['loan_percent_income']),
            'cb_person_cred_hist_length': float(request.form['cb_person_cred_hist_length']),
            'loan_grade_enc': float(request.form['loan_grade_enc']),
            'default_on_file': float(request.form['default_on_file']),
            'home_RENT': 1 if request.form['person_home_ownership'] == 'RENT' else 0,
            'home_OWN': 1 if request.form['person_home_ownership'] == 'OWN' else 0,
            'home_MORTGAGE': 1 if request.form['person_home_ownership'] == 'MORTGAGE' else 0,
            'intent_EDUCATION': 1 if request.form['loan_intent'] == 'EDUCATION' else 0,
            'intent_HOMEIMPROVEMENT': 1 if request.form['loan_intent'] == 'HOMEIMPROVEMENT' else 0,
            'intent_MEDICAL': 1 if request.form['loan_intent'] == 'MEDICAL' else 0,
            'intent_PERSONAL': 1 if request.form['loan_intent'] == 'PERSONAL' else 0,
            'intent_VENTURE': 1 if request.form['loan_intent'] == 'VENTURE' else 0,
            'intent_DEBTCONSOLIDATION': 1 if request.form['loan_intent'] == 'DEBTCONSOLIDATION' else 0,
            'loan_to_income': float(request.form['loan_amnt']) / (float(request.form['person_income']) + 1),
            'high_risk_grade': 1 if float(request.form['loan_grade_enc']) >= 5 else 0,
            'high_dti': 1 if float(request.form['loan_percent_income']) > 0.4 else 0,
            'Geography_Russia': 1,
        }

        for col in feature_names:
            if col not in input_data:
                input_data[col] = 0

        df_input = pd.DataFrame([input_data])[feature_names]
        df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])
        probability = model.predict_proba(df_input)[0][1]

    return render_template('index.html', result=probability, accuracy=int(accuracy * 100))


@app.route('/data_info')
def data_info():
    """Страница с информацией о данных, на которых обучалась модель"""
    return render_template('data_info.html', stats=data_stats, feature_names_ru=feature_names_ru)


if __name__ == '__main__':
    print("\n🚀 ЗАПУСК: http://127.0.0.1:5001")
    print("📊 СТРАНИЦА С ДАННЫМИ: http://127.0.0.1:5001/data_info")
    app.run(debug=False, host='0.0.0.0', port=5001)