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

# ---- Benchmark targets ----

benchmark-generate:
	python3 -m benchmarks.src.generate_synthetic_dataset --out benchmarks/dataset --n 50 --seed 42

benchmark-run:
	python3 -m benchmarks.src.run_benchmarks --config benchmarks/config/benchmark.yaml --augment --report --mock-llm

benchmark-report:
	python3 -m benchmarks.src.report --csv benchmarks/results/axiomesg_benchmark_runs.csv --config benchmarks/config/benchmark.yaml

benchmark-test:
	python3 -m pytest backend/tests benchmarks/tests -v

benchmark-all: benchmark-generate benchmark-run benchmark-report benchmark-test
	@echo "✅ Full benchmark pipeline complete."

# ---- No-Filter Benchmark targets ----

benchmark-no-filter-generate:
	python3 -m benchmarks.src.generate_synthetic_dataset --out benchmarks/dataset --n 50 --seed 42

benchmark-no-filter-run:
	python3 -m benchmarks.src.run_benchmarks --config benchmarks/config/benchmark.yaml --benchmark-version no_filter_v2 --disable-pre-algorithm-filter --mock-llm --report

benchmark-no-filter-report:
	python3 -m benchmarks.src.report --csv benchmarks/results/no_filter_benchmark/axiomesg_no_filter_benchmark_runs.csv --config benchmarks/config/benchmark.yaml

benchmark-no-filter-all: benchmark-no-filter-generate benchmark-no-filter-run
	@echo "✅ Full No-Filter benchmark pipeline complete."
