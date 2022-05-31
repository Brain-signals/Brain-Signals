default:
	@echo "Please specify a make command. help command might be usefull"

help:
	@echo "\nHelp for the VAPE-MRI project package Makefile"

	@echo "\n  help"
	@echo "    Show this help"

	@echo "\n  reinstall_package"
	@echo "    Uninstall and reinstall VAPE-MRI virtual env, and its requirements"


reinstall_package:
	@pip uninstall -y VAPE-MRI || :
	@pip install -e .

hard_uninstall:
	@pip uninstall -yr requirements.txt VAPE-MRI

choose_GCP_project:
	@gcloud config set project ${PROJECT}
	@echo "Chosen GCP project : ${PROJECT}"
