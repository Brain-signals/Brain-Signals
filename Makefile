
default:
	@echo "Please specify a make command. help command might be usefull"

reinstall_package:
	@pip uninstall -y brain-signals && pip install -e .

hard_uninstall:
	@pip uninstall -yr requirements.txt brain-signals

train_model:
	@python -m brainsignals.main

dl_datasets:
	@gsutil -m cp -ncr gs://vape-mri/processed_datasets ${DATASETS_PATH}/..

update_registry:
	@gsutil -m cp -nc ${LOCAL_REGISTRY_PATH}/* gs://vape-mri/registry || :
	@gsutil -m cp -nc gs://vape-mri/registry/* ${LOCAL_REGISTRY_PATH}

run_api:
	@uvicorn brainsignals.api:app --reload

help:
	@echo "\nHelp for the VAPE-MRI project package Makefile"

	@echo "\n  help"
	@echo "    Show this help"

	@echo "\n  reinstall_package"
	@echo "    Uninstall and reinstall VAPE-MRI virtual env, and its requirements"
	@echo "    to completely uninstall VAPE-MRI, and its requirements, use hard_uninstall"

	@echo ""
