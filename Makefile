.ONESHELL:
.PHONY: help
help: ## Print this help message.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

.PHONY: test
test: ## Run all unittests.
	python -m unittest discover -s tests

# Example workflow
EXP_DIR := experiments/example
PARAM_FILES := $(wildcard $(EXP_DIR)/params/*.yaml)
RESULT_FILES := $(patsubst $(EXP_DIR)/params/%.yaml,$(EXP_DIR)/results/%.yaml,$(PARAM_FILES))

PLOT_DIR := reports/example/plots
PLOT_FILES := $(PLOT_DIR)/Test_surface_dof.pdf \
              $(PLOT_DIR)/Test_surface_width.pdf \
              $(PLOT_DIR)/Train_surface_dof.pdf \
              $(PLOT_DIR)/Train_surface_width.pdf

example: ## Run complete example workflow
	@if [ -z "$$(ls $(EXP_DIR)/params/*.yaml 2>/dev/null)" ]; then \
		cd $(EXP_DIR) && python makeparams.py; \
	fi
	$(MAKE) plots

# Generate parameters
$(EXP_DIR)/params/%.yaml: $(EXP_DIR)/makeparams.py
	cd $(EXP_DIR) && python makeparams.py

# Run each individual training
$(EXP_DIR)/results/%.yaml: $(EXP_DIR)/params/%.yaml
	cd $(EXP_DIR) && python runner.py params/$*.yaml

# Collect results into a dataframe
$(EXP_DIR)/results/df.pkl: $(RESULT_FILES) $(EXP_DIR)/dataframe.py
	cd $(EXP_DIR) && python dataframe.py

plots: $(PLOT_FILES) ## Generate all surface plots

# Make plots
$(PLOT_FILES): $(EXP_DIR)/results/df.pkl $(EXP_DIR)/plotter.py
	cd $(EXP_DIR) && python plotter.py