Run the main.py
gameConfig.py --> contains a config file (for different games, you just create new dataclasses. This saves configs)
games.py --> BaseGameStructure.py --> create a class for this and overwrite the functions to create different varities.
main.py --> runs the LLM Games and stores everything properly in a df till now. (CoT integration is pending).