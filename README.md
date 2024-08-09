# RL con data tabular

## Usage

Para entrenar se usa el archivo `agent_model_train.py`

```
python agent_model_train.py [MODEL] [-t | -s [PATH_TO_MODEL]]
```

Para entrenar se ocupa `-t` y para predecir `-s [PATH_TO_MODEL]`

Ejemplo

```
python agent_model_train.py DQN -t
```

## Funcionamiento básico de sb3

El flujo que ocupa sb3 para su entrenamiento es ocupar la función train, que sigue el siguiente flujo:

1. `reset()` para el primer elemento de un split
2. `step()` para los siguientes hasta terminar el split dada la variable `terminated`

y así sucesivamente hasta que se detiene el entrenamiento.

Estas dos funciones se encuentran en el archivo `agent_model_env.py`
