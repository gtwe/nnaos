.PHONY: help
help: ## Print this help message.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

.PHONY: test
test: ## Run all unittests.
	python -m unittest discover -s tests
