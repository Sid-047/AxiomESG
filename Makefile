dev:
	docker-compose up

backend:
	cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm install && npm run dev
