#install:
#	pip install --upgrade pip &&\
#		pip install -r requirements.txt

format:	
	black *.py 

main:
	python run.py

localstack:
	docker-compose up --build

s3:
	python upload_s3.py

deploy:
	http://localhost:8501

monitor:
	http://localhost:8502


	
