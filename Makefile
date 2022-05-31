help:
	@echo "\nHelp for the VAPE-MRI project package Makefile\n"

reinstall_package:
	@pip uninstall -y taxifare-model || :
	@pip install -e .
