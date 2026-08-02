# 🛍️ Smart Retail & Customer Intelligence Platform

An AI-powered retail platform that recognizes returning customers via face
recognition, classifies product images, analyzes customer feedback sentiment,
answers FAQs through a chatbot, and exposes everything through a single
production-style FastAPI gateway.

This project maps every module to a concrete deliverable:

| Syllabus Topic | Project Module | Location |
|---|---|---|
| OpenCV basics | Image preprocessing, webcam/video capture | `app/services/cv_utils.py` |
| Image classification | Product category classifier (MobileNetV2) | `notebooks/train_image_classifier.py`, `app/models/product_classifier.h5` |
| Face recognition | Customer recognition & visit logging (LBPH) | `notebooks/setup_face_recognition.py`, `app/models/face_db.pkl` |
| Text preprocessing | Cleaning reviews/chat text | `notebooks/train_sentiment_model.py` (`clean_text`) |
| Sentiment analysis | Customer review/feedback classifier | `app/models/sentiment_model.pkl`, `app/models/vectorizer.pkl` |
| Chatbot basics | FAQ/support chatbot (rule-based + ML) | `app/services/chatbot_service.py`, `app/models/chatbot_model.pkl` |
| ML pipelines | Unified pipeline combining all models | `app/main.py`, `app/services/*.py` |
| Pickle / Joblib | Model serialization | all `.pkl` files in `app/models/` |
| Flask / FastAPI | REST API serving all models | `app/main.py`, `app/routers/*.py` |
| API deployment | Dockerized deployment to cloud | `Dockerfile`, `.github/workflows/deploy.yml` |

---

## 1. System Architecture

```
Client Layer (dashboard / Postman / webcam feed)
        |  REST calls
        v
FastAPI Gateway
  /recognize-face   /classify-product
  /analyze-sentiment   /chatbot   /dashboard/stats
        |
  -------------------------------------------
  |                 |                        |
  v                 v                        v
CV Module        NLP Module              Chatbot Module
- OpenCV capture  - Text cleaning         - Intent matching
- Face embeddings - TF-IDF / BERT         - Rule + ML hybrid
- CNN classifier  - Sentiment model       - FAQ retrieval
        |                 |                        |
        -------------------------------------------
                        |
                        v
  Storage: model files + in-memory logs
  (customer_visits, reviews, chat_logs, model.pkl/.h5)
```

Every model is loaded **once at startup** (see the top-level imports in
`app/services/cv_service.py`, `nlp_service.py`, `chatbot_service.py`) and
reused across all requests — this is the "unified pipeline" (Module C1).

---

## 2. Project Structure

```
smart-retail-ai/
├── app/
│   ├── main.py                # FastAPI entrypoint
│   ├── routers/
│   │   ├── vision.py
│   │   ├── nlp.py
│   │   └── chatbot.py
│   ├── models/
│   │   ├── product_classifier.h5
│   │   ├── face_db.pkl
│   │   ├── sentiment_model.pkl
│   │   └── chatbot_model.pkl
│   ├── services/
│   │   ├── cv_utils.py         # OpenCV basics (Module A1)
│   │   ├── cv_service.py
│   │   ├── nlp_service.py
│   │   └── chatbot_service.py
│   └── schemas.py
├── notebooks/                 # training/experiment notebooks
│   ├── 01_image_classifier_training.ipynb
│   ├── 02_face_recognition_setup.ipynb
│   └── 03_sentiment_model_training.ipynb
├── data/
│   ├── reviews.csv
│   └── intents.json
├── tests/
│   └── test_endpoints.py
├── requirements.txt
├── Dockerfile
├── README.md
└── .github/workflows/deploy.yml
```

---

## 3. Quickstart (local)

```bash
git clone https://github.com/<your-username>/smart-retail-ai.git
cd smart-retail-ai

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Set your API key (or rely on the dev default)
export API_KEY="dev-demo-key-change-me"    # Windows: set API_KEY=dev-demo-key-change-me

uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the auto-generated Swagger UI.

Every request needs the header `X-API-Key: dev-demo-key-change-me` (Module C4:
simulated production security). Change `API_KEY` before deploying anywhere real.

### Example calls

```bash
curl -X POST http://localhost:8000/analyze-sentiment \
  -H "X-API-Key: dev-demo-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"text": "The shoes I ordered are amazing!"}'

curl -X POST http://localhost:8000/chatbot \
  -H "X-API-Key: dev-demo-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is your return policy?"}'

curl -X POST http://localhost:8000/classify-product \
  -H "X-API-Key: dev-demo-key-change-me" \
  -F "image=@/path/to/product.jpg"

curl -X POST http://localhost:8000/recognize-face \
  -H "X-API-Key: dev-demo-key-change-me" \
  -F "image=@/path/to/face.jpg"

curl http://localhost:8000/dashboard/stats -H "X-API-Key: dev-demo-key-change-me"
```

---

## 4. Retraining the models

Trained model artifacts already ship in `app/models/` so the API runs
out of the box. To retrain any of them:

```bash
python notebooks/train_image_classifier.py     # -> product_classifier.h5
python notebooks/setup_face_recognition.py     # -> face_db.pkl
python notebooks/train_sentiment_model.py      # -> sentiment_model.pkl, vectorizer.pkl
python notebooks/train_chatbot_model.py        # -> chatbot_model.pkl
```

> **Important — swap in real data for a real deployment.** To keep this
> repo self-contained (no external downloads required), the image
> classifier and face recognizer train on small **synthetic placeholder
> datasets** generated in code. For production-quality accuracy:
> - Image classifier: point `generate_synthetic_dataset()` in
>   `train_image_classifier.py` at a real dataset such as the Kaggle
>   **[Retail Product Checkout Dataset](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset)**
>   or **Fashion-MNIST**, and set `weights="imagenet"` on `MobileNetV2` if
>   you have internet access at train time.
> - Face recognition: replace `generate_synthetic_faces()` with real
>   (consenting) customer photos organized as `data/faces/<name>/*.jpg`,
>   or the **[Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)**
>   dataset for practice.
> - Sentiment: `data/reviews.csv` ships with 220 synthetic labeled
>   reviews; swap in the Kaggle
>   **["Women's E-Commerce Clothing Reviews"](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)**
>   dataset (or Amazon Reviews) for real-world accuracy, keeping the same
>   `review_text` / `sentiment` column names.
> - Chatbot: `data/intents.json` ships with 25 custom FAQ intents; extend
>   it with your own patterns/responses as your product catalog grows.

---

## 5. Testing

```bash
pytest tests/ -v
```

14 tests cover: root/health checks, API-key auth (missing/invalid), sentiment
analysis (positive/negative/validation), chatbot (rule-based + fallback),
product classification (success + bad content-type), face recognition
(no-face case), and dashboard stats.

CI (`.github/workflows/deploy.yml`) runs `flake8` + `pytest` on every push/PR,
then builds the Docker image.

---

## 6. Deployment (Docker)

```bash
docker build -t smart-retail-ai .
docker run -p 8000:8000 -e API_KEY=your-production-key smart-retail-ai
```

Push the image to any container registry and deploy it to:
- **Render** or **Railway** (free-tier friendly, connect the GitHub repo directly)
- **Google Cloud Run** (`gcloud run deploy --image <your-image>`)
- **AWS EC2 / ECS**

Add a `RENDER_DEPLOY_HOOK` / `GCP` service-account secret to the repo and
extend `.github/workflows/deploy.yml`'s `build-image` job to push + trigger
your host's deploy step for full CI/CD.

---

## 7. Ethics & Privacy Note (Face Recognition)

This project's face recognition module is built for a **loyalty-analytics
use case** ("has this customer visited before?") and ships with a synthetic
demo dataset, not real biometric data. A production deployment must address:

- **Consent** — customers should be clearly informed and opt in before their
  face is enrolled; opt-out and deletion mechanisms are required.
- **Data privacy & retention** — face encodings are sensitive biometric data
  under regulations like GDPR/CCPA and India's DPDP Act; store them encrypted,
  minimize retention, and restrict access.
- **Bias** — face recognition models can have uneven accuracy across
  demographic groups; any real deployment should be audited for bias before
  and during rollout, with a human-review fallback for low-confidence matches.
- **Transparency** — signage/notices at store entrances, and a documented
  policy on how visit data is used (e.g., loyalty analytics only, not sold
  to third parties).

---

## 8. Stretch Goals

- Fine-tune a **DistilBERT** sentiment model (via HuggingFace `transformers`)
  in place of the TF-IDF + Logistic Regression baseline.
- Add a **WebSocket** endpoint for real-time video-stream face recognition.
- Log prediction confidence over time to monitor **model/data drift**.
- Build a **Streamlit** dashboard on top of `GET /dashboard/stats`.
- A/B test two chatbot response strategies and log which gets rated higher.

---

## 9. License

MIT — free to use and adapt for coursework, portfolios, or as a starting
point for a real deployment.
