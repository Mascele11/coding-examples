# Intro

- Be sure you are in the correct directory, otherwise `cd src/sampleapplication/`
- Make sure you have python 3.X or greater (check it using `python -V`)
- Then create the virtual environment. It should be specific for this project and placed inside its directory (so to be cleaned together with the repo)
    - venv: `python3.X -m venv ./venv-package/`
    - conda: `conda create --prefix ./conda-package/ python=3.7`
- Activate your virtual environment
    - venv:
        - UNIX `source venv-package/bin/activate`
        - WS `venv-package\Scripts\activate`
    - conda:
        - UNIX `conda activate ./conda-package/`
        - WS `conda activate .\conda-package\ `
- Install python dependencies `pip install -r requirements`

