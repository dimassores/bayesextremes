IMAGE=bayesextremes:latest

build:
	docker build . -t $(IMAGE)

run:
	docker run \
		-p 8888:8888 \
		-v $(CURDIR):/app \
		$(IMAGE)
