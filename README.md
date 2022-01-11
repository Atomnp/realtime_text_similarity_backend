# realtime_text_similarity_backend
Backend for realtime text semantic similarity identification

# How to run?
1. set git hooks path
    `git config core.hooksPath .githooks`
2. start flask server

3. post request in localhost:5000 with json body which contains "question"
```
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"question":"who is the father of economics?"}' \
  http://localhost:5000
  ```
