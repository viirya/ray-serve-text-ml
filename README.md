# ray-serve-text-ml

This repo contains docker file, YAML definition and python code:

* serve [t5-small](https://huggingface.co/google-t5/t5-small) model on K8s using RayServe
* finetune t5-small with text-to-SQL dataset [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context?row=3) on K8s using RayJob

A huggingface <img src="https://huggingface.co/front/assets/huggingface_logo.svg" width="20" height="20"> model is provided at: [t5-small-text-to-sql](https://huggingface.co/viirya/t5-small-text-to-sql)

## Download model

For isolated K8s, the serving application cannot download the model from huggingface at runtime.
Instead, we can download the model files locally and put into the docker image.

```
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
>>> from transformers import pipeline

# download model and save it into local disk
>>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
>>> tokenizer.save_pretrained("./t5-small")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
>>> model.save_pretrained("./t5-small")

# test saved model
>>> m = pipeline("translation_en_to_fr", model = "./t5-small")
>>> m("How are you")
[{'translation_text': 'Comment êtes-vous'}]
```

## Build and publish docker image

Note: replace image tag with your docker registry and username.
```
docker build -f Dockerfile -t registry-host/username/ray:2.9.0-py39-text-ml-v3 .
docker push registry-host/username/ray:2.9.0-py39-text-ml-v3
```

## Deploy to K8s

Suppose `KubeRay` controller is already installed in K8s cluster.

```
kubectl apply -f ray-service.text-ml.yaml --namespace ray-test  
```

Check if rayservice is deployed and running:

```
kubectl describe rayservices/rayservice-sample --namespace ray-test
```

If everything is running okay, you will see:

```
  Normal  Running                      11s (x434 over 14m)  rayservice-controller  The Serve applicaton is now running and healthy.
```

You can then see the service in K8s:

```
kubectl get services --namespace ray-test
```

```
rayservice-sample-serve-svc                          ClusterIP   10.100.88.94     <none>        8000/TCP                                        16m
```

Creating port forwarding:

```
kubectl port-forward service/rayservice-sample-serve-svc 8000 --namespace ray-test
```

Query the text-ml service:
```
curl -X POST -H "Content-Type: application/json" localhost:8000/summarize_translate -d '"It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief"'
```

You should get the response:
```
c'était le meilleur des temps, c'était le pire des temps .%                                                                                                 ```                                                                           
