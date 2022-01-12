# realtime_text_similarity_backend

Backend for realtime text semantic similarity identification

## How to run?

1. set git hooks path

    `git config core.hooksPath .githooks` --did not work

    `git config --unset-all core.hooksPath`
    `pre-commit install`

2. start flask server

    ```bash
    pip install -r requirements.txt
    ./run.sh
    ```

    ```powershell
    pip install -r requirements.txt
    ./run.ps1
    ```

3. post request in localhost:5000 with json body which contains "question"

    ```bash
    curl --header "Content-Type: application/json" \
      --request POST \
      --data '{"question":"who is the father of economics?"}' \
      http://localhost:5000
    ```

    ```powershell
    Invoke-WebRequest -UseBasicParsing http://localhost:5000 -ContentType "application/json" -Method POST -Body '{ "question": "who is the father of economics?" }'
    ```
