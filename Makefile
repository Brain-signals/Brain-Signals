

default:
	@echo "Please specify a make command. help command might be usefull"

help:
	@echo "\nHelp for the VAPE-MRI project package Makefile"

	@echo "\n  help"
	@echo "    Show this help"

	@echo "\n  reinstall_package"
	@echo "    Uninstall and reinstall VAPE-MRI virtual env, and its requirements"
	@echo "    to completely uninstall VAPE-MRI, and its requirements, use hard_uninstall"

	@echo "\n  GCP_select"
	@echo "    select the gcloud project set in .env as PROJECT"

	@echo "\n  GCP_start"
	@echo "    start the GCE machine"

	@echo "\n  GCP_stop"
	@echo "    clean stop the GCE machine"

	@echo "\n  GCP_connect"
	@echo "    ssh connect to the GCE machine"

reinstall_package:
	@pip uninstall -y VAPE-MRI || :
	@pip install -e .

hard_uninstall:
	@pip uninstall -yr requirements.txt VAPE-MRI

GCP_select:
	@gcloud config set project ${PROJECT}
	@echo "Chosen GCP project : ${PROJECT}"

GCP_start:
	@gcloud compute instances start ${VM_INSTANCE}

GCP_stop:
	@gcloud compute instances stop ${VM_INSTANCE}

GCP_connect:
	@gcloud compute ssh ${VM_INSTANCE}

#DL_datasets:
#	@gsutil -m cp -r gs://vape-mri/processed_datasets/ ${DATASETS_PATH}

# DL_datasets:
# 	@echo "This command will delete all files in ${DATASETS_PATH}"
# 	@echo "And download the complete vape-mri dataset from google cloud storage\n"

# 	@read -p "Please confirm : Y/n ? " -n 1 -r
# 	@if ((${REPLY} = ${CONFIRM})); then \
# 		echo "yes"; \
# 	else \
# 		echo "no"; \
# 	fi
