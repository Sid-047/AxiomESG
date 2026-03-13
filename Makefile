dev:
	docker compose up --build

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

restart:
	docker compose restart

build:
	docker compose build --no-cache

test:
	docker exec -it axiomesg-backend python -m pytest tests/ -v

backend:
	cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm install && npm run dev

clean:
	docker compose down --rmi all --volumes --remove-orphans
