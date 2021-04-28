load_data:
	@echo "------------- loading Data -------------"
	wget http://216.47.152.173:30001/nilm/data.zip
	unzip data.zip
	rm data.zip*

clean_data:
	rm -r data/