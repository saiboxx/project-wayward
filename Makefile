PYTHON_INTERPRETER = python

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Start training an agent
train-ddpg:
	$(PYTHON_INTERPRETER) src/ddpg/main_unity.py
train-ppo:
	$(PYTHON_INTERPRETER) src/ppo/main_unity.py

## Start hyperparameter optimization
opt-ppo:
	$(PYTHON_INTERPRETER) src/ppo/optimizer.py
opt-ddpg:
	$(PYTHON_INTERPRETER) src/ddpg/optimizer.py

## Start training an agent in a gym dummy environment
mock-ddpg:
	$(PYTHON_INTERPRETER) src/ddpg/main_gym.py
mock-ppo:
	$(PYTHON_INTERPRETER) src/ppo/main_gym.py

## Load a model and observe agent
run-ddpg:
	$(PYTHON_INTERPRETER) src/ddpg/run.py
run-ppo:
	$(PYTHON_INTERPRETER) src/ppo/run.py

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif