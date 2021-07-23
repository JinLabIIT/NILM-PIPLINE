load_data:
	@echo "------------- loading Data -------------"
	wget http://216.47.152.173:30001/nilm/data.zip
	unzip data.zip
	rm data.zip*
	
	@echo "------------- loading Model Weight -------------"
	wget http://216.47.152.173:30001/nilm/model_weights.zip
	unzip model_weights.zip -d ./experiment/
	rm model_weights.zip*
	

clean_data:
	rm -r data/ experiment/model_weights/
