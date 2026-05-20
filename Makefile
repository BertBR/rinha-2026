UPSTREAM := https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main
DATA := data

.PHONY: help data deps install build-index dev up down logs bench clean smoke verify-detect

help:
	@echo "make deps         install npm deps locally (for dev / smoke tests)"
	@echo "make data         fetch upstream reference and test files into data/"
	@echo "make build-index  run the index builder locally (needs make data first)"
	@echo "make up           docker compose up (full stack, builds index inside)"
	@echo "make down         docker compose down"
	@echo "make logs         tail compose logs"
	@echo "make bench        run k6 against the running stack on :9999"
	@echo "make smoke        in-process smoke test on a small reference subset"
	@echo "make verify-detect  validate 0 FP / 0 FN against test-data.json"
	@echo "make clean        remove generated artifacts"

deps:
	npm install

data: $(DATA)/references.json.gz $(DATA)/mcc_risk.json $(DATA)/normalization.json $(DATA)/example-payloads.json $(DATA)/test-data.json $(DATA)/example-references.json

$(DATA)/references.json.gz:
	@mkdir -p $(DATA)
	curl -sSL -o $@ $(UPSTREAM)/resources/references.json.gz

$(DATA)/mcc_risk.json:
	@mkdir -p $(DATA)
	curl -sSL -o $@ $(UPSTREAM)/resources/mcc_risk.json

$(DATA)/normalization.json:
	@mkdir -p $(DATA)
	curl -sSL -o $@ $(UPSTREAM)/resources/normalization.json

$(DATA)/example-payloads.json:
	@mkdir -p $(DATA)
	curl -sSL -o $@ $(UPSTREAM)/resources/example-payloads.json

$(DATA)/example-references.json:
	@mkdir -p $(DATA)
	curl -sSL -o $@ $(UPSTREAM)/resources/example-references.json

$(DATA)/test-data.json:
	@mkdir -p $(DATA)
	curl -sSL -o $@ $(UPSTREAM)/test/test-data.json

build-index: data
	DATA_DIR=./data node --experimental-strip-types --max-old-space-size=4096 src/build-index.ts

up: data
	docker compose up --build

down:
	docker compose down -v

logs:
	docker compose logs -f --tail=100

bench:
	k6 run bench/k6.js

smoke:
	DATA_DIR=./data node --experimental-strip-types src/test/smoke.ts

verify-detect:
	DATA_DIR=./data node --experimental-strip-types src/test/verify-detection.ts

clean:
	rm -rf node_modules
	rm -f data/*.bin
