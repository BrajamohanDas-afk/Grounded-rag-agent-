.PHONY: install dev test ingest ask chat api ui eval eval-compare graph docker clean

install:            ## core + Anthropic + local embeddings + UI
	pip install -e ".[anthropic,local,ui,dev]"

install-all:
	pip install -e ".[all]"

test:               ## offline test-suite (no API key needed)
	pytest -q

ingest:
	grag ingest data/docs

ask:                ## make ask Q="your question"
	grag ask "$(Q)" -v

chat:
	grag chat -v

api:
	uvicorn grounded_rag.api:app --reload --port 8000

ui:
	streamlit run app/streamlit_app.py

graph:              ## print the agent graph as Mermaid
	grag graph > docs/graph.mmd && cat docs/graph.mmd

eval:
	python eval/run_eval.py --preset full

eval-compare:       ## ablation study across presets
	python eval/run_eval.py --compare

eval-retrieval:
	python eval/run_eval.py --retrieval-only

docker:
	docker compose up --build

clean:
	rm -rf index eval/results .pytest_cache
