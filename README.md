# 📝 Automatic Essay Evaluation

A system for automatic essay evaluation using NLP and machine learning.

---

## 🛠 Installation

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/TimonLi4/EssayEvaluation
```

### 2️⃣ Install Dependencies
```sh
pip install -r req.txt
```

### 3️⃣ Install Additional Resources

#### 🔹 Install `spaCy` Model
Run the following command in the console:
```sh
python -m spacy download en_core_web_sm
```

#### 🔹 Download NLTK Resources
Run the `download.py` file to install required NLTK resources.

### 4️⃣ Set Up API Key
- Generate an API key at [Together AI](https://api.together.ai/).
- Paste the API key into the `.env` file located at:
  ```
  EssayEvaluation\AutomaticEssayEvaluation\AutomaticEssayEvaluation\.env
  ```

### 5️⃣ Run the Server
Navigate to the `manage.py` directory:
```sh
cd EssayEvaluation\AutomaticEssayEvaluation
```
Then start the server:
```sh
python manage.py runserver
```

Now you can select a file for evaluation. 🚀

