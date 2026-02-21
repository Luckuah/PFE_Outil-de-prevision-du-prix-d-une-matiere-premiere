# PFE_Outil-de-prevision-du-prix-d-une-matiere-premiere

To get the project run:

```bash
git clone https://github.com/Luckuah/PFE_Outil-de-prevision-du-prix-d-une-matiere-premiere.git
cd PFE_Outil-de-prevision-du-prix-d-une-matiere-premiere
```

To initialize the dependencies:

```bash
poetry install
```

To create the env:

```bash
poetry run python -m ipykernel install --user --name=pfe-poetry --display-name "PFE Poetry Env"
```

To launch The API:

**IMPORTANT 1: Before running the API you need to create your SQL databse with gdelt.sql in project_file\Pipeline_Data\database**

**IMPORTANT 2: You need to lauch the API and wait for the console return (INFO: Application startup complete.) to run the App (it can take some time)**

```bash
poetry run uvicorn API:app --reload
```

To run the App:

```bash
poetry run streamlit run main.py
```

