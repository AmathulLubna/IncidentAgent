# MemorySRE React frontend

A production-style React/Vite control room for the Incident Response Agent.

## Run

```bash
cd frontend
npm install
npm run dev
```

The frontend defaults to `http://localhost:8000` for the FastAPI backend.
To override it:

```bash
# frontend/.env.local
VITE_API_URL=http://localhost:8000
```

Open the Vite URL (normally `http://localhost:5173`).
