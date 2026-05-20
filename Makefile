UPSTREAM := https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main
DATA := data

.PHONY: help data clean go-up go-down node-up node-down bench

help:
	@echo "make data       fetch reference files from upstream"
	@echo "make go-up      docker compose up Go submission"
	@echo "make node-up    docker compose up Node submission"
	@echo "make bench      run k6 against running stack"
	@echo "make clean      remove generated artifacts"

data: $(DATA)/references.json.gz $(DATA)/mcc_risk.json $(DATA)/normalization.json $(DATA)/example-payloads.json $(DATA)/test-data.json

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

$(DATA)/test-data.json:
	@mkdir -p $(DATA)
	curl -sSL -o $@ $(UPSTREAM)/test/test-data.json

go-up: data
	docker compose -f docker-compose.go.yml up --build

go-down:
	docker compose -f docker-compose.go.yml down

node-up: data
	docker compose -f docker-compose.node.yml up --build

node-down:
	docker compose -f docker-compose.node.yml down

bench:
	k6 run bench/k6.js

clean:
	rm -rf go/bin go/dist node/dist data/*.bin data/*.index
