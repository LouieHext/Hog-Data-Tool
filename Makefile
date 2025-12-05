spellcheck:  ## Run spellcheck
	codespell --ignore-words=.codespell/ignore.txt --skip="tags,*_build,./htmlcov/*,./data/*,./dist/*,./.venv/*,./venv/*"


lint: spellcheck ## Run lint steps (ruff, black, pyright)
	RC=0; \
	ruff --fix --preview --show-fixes --output-format=full . || RC=1; \
	black . || RC=1; \
	pyright . || RC=1; \
	exit $$RC

run:
	poetry run python -m hog_data_tool.run

