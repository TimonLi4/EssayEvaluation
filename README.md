# 📝 Automatic Essay Evaluation

A system for automatic essay evaluation using NLP and machine learning.

---

## 🛠 Installation

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/TimonLi4/EssayEvaluation
```

### 2️⃣ Create a Virtual Environment
Navigate to the project directory and create a virtual environment:
```sh
python -m venv .venv
```
Activate the virtual environment:
- **Windows:**
  ```sh
  .venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```sh
  source .venv/bin/activate
  ```

### 3️⃣ Install Dependencies
follow the path:
```sh
cd EssayEvaluation\AutomaticEssayEvaluation
```
Then install dependencies:
```sh
pip install -r req.txt
```

### 4️⃣ Install Additional Resources

#### 🔹 Download NLTK Resources
Run the `download.py` file to install required NLTK resources.

### 5️⃣ Set Up API Key
- Generate an API key at [Together AI](https://api.together.ai/).
- Paste the API key into the `.env` file located at:
  ```
  EssayEvaluation\AutomaticEssayEvaluation\AutomaticEssayEvaluation\.env
  ```

### 6️⃣ Run the Server
```sh
python manage.py runserver
```

Now you can select a file for evaluation. 🚀

