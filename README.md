
```markdown
# Fraud Detection Retriever Agent

This project implements a **retriever agent** that finds the most similar past credit applications for a given new application. It uses:

- **PostgreSQL + pgvector** to store and search feature vectors  
- **HNSW indexing** for fast approximate nearest neighbor search  
- **Flask** as a REST API (A2A‑compatible endpoint)  
- A preprocessing pipeline (scaling, one‑hot encoding, missing value handling)  

The agent returns the top‑k similar cases, their fraud labels, and a local fraud rate to help assess risk.

---

## 📦 Requirements

- **Docker** (to run the PostgreSQL container)  
- **Python 3.9+** with packages: `pandas`, `numpy`, `psycopg2`, `flask`, `joblib`, `scikit-learn`, `jupyter`  
- At least **8 GB RAM** (16 GB recommended for 1 million rows)  
- `base.csv` – the BAF dataset (1 million rows, 32 columns) – **not included in this repository** (place it in the project folder)

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/claranatalie01/PROJECT.git
cd PROJECT
```

### 2. Start the PostgreSQL + pgvector container

```bash
docker run --name postgres -p 5433:5432 -e POSTGRES_PASSWORD=mysecretpassword --shm-size=2g -d pgvector/pgvector:pg16
```

Check that it’s running:

```bash
docker ps
```

> **Port note:** The container maps host port `5433` to container port `5432`. This avoids conflicts if you already have a local PostgreSQL on port `5432`.

---

## 🧹 Data Preprocessing & Loading

The raw dataset must be transformed into feature vectors and inserted into the database.  
**All of this is done in a single Jupyter notebook:** `preprocess_load_data.ipynb`.

### Run the notebook

```bash
jupyter notebook preprocess_load_data.ipynb
```

Execute all cells **in order**. The notebook will:

- Read `base.csv`
- Handle missing values (replace `-1` with `NaN`, then fill with medians)
- One‑hot encode categorical columns
- Scale all features using `StandardScaler`
- Save the fitted preprocessors (`scaler.pkl`, `encoder.pkl`, `medians.pkl`, `feature_cols.pkl`)
- Load the preprocessed vectors into the PostgreSQL table `applications`

*Expected output at the end:*
```
⏳ Inserting data... (may take several minutes)
✅ Data insertion complete.
```

### Verify the data

```bash
docker exec -it postgres psql -U postgres -c "SELECT COUNT(*) FROM applications;"
```

Should return `1000000`.

---

## ⚡ Create Indexes for Fast Search

Connect to the database:

```bash
docker exec -it postgres psql -U postgres
```

Run the following SQL commands **inside the psql prompt**:

```sql
SET maintenance_work_mem = '2GB';
CREATE INDEX ON applications USING hnsw (feature_vector vector_cosine_ops);
CREATE INDEX ON applications (month);
```

Check that both indexes exist:

```sql
\di
```

Exit psql:

```sql
\q
```

> **Why indexes?** The HNSW index makes cosine similarity searches sub‑second even with millions of vectors. The `month` index speeds up filtering by month.

---

## 🚀 Start the Retriever Agent (Flask API)

The agent is implemented in `A2A.py`. It loads the preprocessors (the `.pkl` files), connects to the database, and serves a `/agent/retriever` endpoint.

```bash
python A2A.py
```

You should see:

```
 * Running on http://0.0.0.0:5001
```

Leave this terminal open. The agent is now ready to accept requests.

---

## 🧪 Testing the Agent

### Option 1: Use the provided test script

`test_agent.py` contains a sample application (you can replace the `query` with your own data).

```bash
python test_agent.py
```

Expected output (abbreviated):

```json
{
  "task_id": "test-001",
  "output": {
    "similar_cases": [...],
    "local_fraud_rate": 0.6,
    "total_neighbors": 5
  }
}
```

### Option 2: Find real fraudulent cases as test queries

The script `find.py` scans a random sample of fraudulent applications, computes the local fraud rate among their nearest neighbors (from strictly earlier months), and prints the IDs with the highest rates. You can then copy the metadata of the top candidate into `test_agent.py`.

```bash
python find.py
```

*Example output:*
```
=== Top 10 fraudulent applications with highest local fraud rate ===
 id  month  fraud_neighbors  total_neighbors  local_fraud_rate
123   6              95               100              0.95
...

Metadata for this application (copy into test.py):
{
  "month": 6,
  "income": 0.42,
  ...
}
```

Copy the entire `metadata` object into the `"query"` field of `test_agent.py` and re‑run it to see how the agent responds to a known fraudulent case.

---

## 🔁 Using the API programmatically

Send a POST request to `http://localhost:5001/agent/retriever` with the following JSON format:

```json
{
  "id": "request-123",
  "input": {
    "query": { ... all application features ... }
  }
}
```

The response contains `similar_cases` (up to 20 nearest neighbors, each with `id`, `fraud_bool`, `month`, `similarity`, `metadata`), `local_fraud_rate`, and `total_neighbors`.

---

## 🗑️ Cleaning Up

Stop and remove the container when you’re done:

```bash
docker stop postgres
docker rm postgres
```

---

## 📁 Repository Structure

```
.
├── base.csv                       # (not committed – large file)
├── preprocess_load_data.ipynb     # Combined preprocessing + loading
├── A2A.py                         # Flask retriever agent
├── test_agent.py                  # Example test request
├── find.py                        # Finds typical fraudulent cases for testing
├── scaler.pkl                     # Generated by notebook (ignored by Git)
├── encoder.pkl
├── medians.pkl
├── feature_cols.pkl
└── README.md
```

---

## ❓ Troubleshooting

| Problem | Likely solution |
|---------|----------------|
| `docker: command not found` | Install Docker Desktop for Mac/Windows |
| `psycopg2` errors | Run `pip install psycopg2-binary` |
| `Cannot connect to database` | Ensure container is running: `docker ps`. Check port mapping (`5433:5432`). |
| `FileNotFoundError` for `.pkl` files | You must run `preprocess_load_data.ipynb` first – the agent needs those files. |
| Slow queries after loading | Did you create the HNSW index? Run the `CREATE INDEX` commands again. |
| Out of memory | Increase Docker memory (Docker → Preferences → Resources → RAM to 6‑8 GB). Also set `--shm-size=2g` as in the run command. |

---

## 📝 Important Notes

- The agent only returns applications from months **before** the current application’s month to avoid look‑ahead bias.  
- The similarity score is `1 - cosine_distance`, so 1 = identical, 0 = orthogonal.  
- The `.pkl` files (preprocessors) are **not** committed to Git. Every user must regenerate them by running the notebook. This ensures reproducibility.  
- `base.csv` is too large for GitHub; place it manually in the project folder before running the notebook.

---

