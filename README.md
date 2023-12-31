# Proyecto integrador 2023 - Grupo 5 (Maestría de Ciencia de datos y analítica)

## Integrantes

* David Armendáriz
* Camilo Vélez
* David López
* Juan Sebastian Avila

## Requisitos para el ambiente local

Recomendamos altamente usar un entorno Linux o Mac para el desarrollo del proyecto. En este proyecto utilizamos `pyenv` para manejar la versión de python y `pipenv` para manejar las dependencias del proyecto. Si no se desea utilizar `pipenv`, se puede instalar directamente con `pip` utilizando el archivo `requirements.txt`.

* Utilizar `pyenv` para manejar una versión local de python especificada en el archivo `.python-version`. Esto se puede instalar siguiendo las instrucciones en <https://github.com/pyenv/pyenv>

  * Linux: `curl https://pyenv.run | bash`
  * MacOS:

  ```zsh
  brew update
  brew install pyenv
  ```

* Instalar `pipenv` para manejar las dependencias del proyecto. Esto se puede instalar siguiendo las instrucciones en <https://pipenv.pypa.io/en/latest/>
  
  ```terminal
  pip install pipenv
  ```

* Instalar las dependencias del proyecto

  ```terminal
  pipenv install
  ```

## Instalación con `requirements.txt`

Si no se desea utilizar `pipenv` para manejar las dependencias del proyecto, se puede instalar directamente con `pip` utilizando el archivo `requirements.txt`

```terminal
pip install -r requirements.txt
```
