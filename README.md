# AutoML Framework

An open-source, no-code Automated Machine Learning framework with a modern web interface. Upload datasets, explore data visually, preprocess automatically, and train optimized ML models — all from your browser.

## Features

- **Dataset Management** — Upload CSV files, preview data, manage multiple datasets
- **Exploratory Data Analysis (EDA)** — Interactive charts, distribution plots, correlation matrices, outlier detection
- **Adaptive Preprocessing** — Automatic handling of missing values, encoding, scaling, outlier treatment, skewness correction, rare value grouping, and more
- **AutoML Training** — Automated model selection and hyperparameter optimization powered by Optuna
- **Model Comparison** — Side-by-side performance metrics, confusion matrices, feature importance
- **Classification & Regression** — Support for both task types with appropriate metrics
- **Advanced Models** — Scikit-learn classifiers/regressors, XGBoost, LightGBM
- **JSON File Storage** — Zero-dependency storage, no database setup required

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+

### 1. Clone the repository

```bash
git clone https://github.com/chadikam/AutoMl.git
cd AutoMl
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env

# Start the API server
python main.py
```

The API will be available at `http://localhost:8000`.

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

The UI will be available at `http://localhost:5173`.

## Usage

1. **Upload a dataset** — Go to the Datasets page, click "Upload", and select a CSV file
2. **Explore your data** — Click on a dataset to view EDA charts, statistics, and data quality reports
3. **Preprocess** — Configure preprocessing options (or use adaptive defaults) and apply them
4. **Train models** — Navigate to AutoML, select your target column, choose task type, and start training
5. **Compare results** — View trained model performance, feature importance, and predictions

## Example

```python
# You can also use the API directly
import requests

# Upload a dataset
with open("my_data.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/datasets/upload",
        files={"file": f}
    )
dataset_id = response.json()["id"]

# Start AutoML training
response = requests.post(
    f"http://localhost:8000/api/automl/start/{dataset_id}",
    json={
        "target_column": "target",
        "task_type": "classification",
        "n_trials": 50,
        "timeout": 300
    }
)
```

## Project Structure

```
AutoMl/
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── app/
│   │   ├── config.py           # Application settings
│   │   ├── storage.py          # JSON file storage engine
│   │   ├── models/
│   │   │   └── schemas.py      # Pydantic data models
│   │   ├── routes/
│   │   │   ├── datasets.py     # Dataset CRUD & EDA endpoints
│   │   │   ├── models.py       # Trained model endpoints
│   │   │   └── automl.py       # AutoML training endpoints
│   │   ├── services/
│   │   │   ├── automl_engine.py        # Optuna-based AutoML engine
│   │   │   ├── automl_plots.py         # Plot generation
│   │   │   ├── training_manager.py     # Training orchestration
│   │   │   ├── eda_service.py          # EDA analysis service
│   │   │   └── adaptive_preprocessing.py  # Adaptive preprocessing
│   │   └── utils/              # Utility functions
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # Main React app with routing
│   │   ├── pages/              # Dashboard, EDA, Training pages
│   │   ├── components/         # Reusable UI components (shadcn/ui)
│   │   └── utils/api.js        # API client
│   ├── package.json
│   └── vite.config.js
├── AutoMl_Datasets/            # Example datasets for testing
└── README.md
```

## API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for the interactive Swagger UI documentation.

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/datasets/upload` | Upload a CSV dataset |
| GET | `/api/datasets/` | List all datasets |
| GET | `/api/datasets/{id}` | Get dataset details |
| GET | `/api/datasets/{id}/eda` | Get EDA analysis |
| POST | `/api/datasets/{id}/preprocess` | Apply preprocessing |
| POST | `/api/automl/start/{id}` | Start AutoML training |
| GET | `/api/automl/status/{id}` | Check training status |
| GET | `/api/models/` | List trained models |

## Configuration

Edit `backend/.env` to customize:

```env
DATA_DIR=data              # JSON storage directory
UPLOAD_DIR=uploads         # Dataset upload directory
MODELS_DIR=trained_models  # Saved model directory
MAX_UPLOAD_SIZE=52428800   # Max file size (50MB)
HOST=0.0.0.0               # Server host
PORT=8000                   # Server port
```

## Tech Stack

**Backend:** FastAPI, Python, scikit-learn, Optuna, XGBoost, LightGBM, pandas, matplotlib

**Frontend:** React, Vite, TailwindCSS, shadcn/ui, Recharts

**Storage:** JSON file-based (zero external dependencies)

## Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create a feature branch:** `git checkout -b feature/my-feature`
3. **Make your changes** and add tests if applicable
4. **Commit:** `git commit -m "Add my feature"`
5. **Push:** `git push origin feature/my-feature`
6. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 for Python code
- Use ESLint/Prettier for JavaScript/React code
- Write descriptive commit messages
- Add docstrings to new functions and classes
- Test your changes before submitting

## License

This project is open source and available under the [MIT License](LICENSE).
