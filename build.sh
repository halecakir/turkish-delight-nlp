docker login --username halecakir
docker build . -t turkish_nlp
docker tag turkish_nlp:latest halecakir/turkish_nlp_toolkit
docker push halecakir/turkish_nlp_toolkit