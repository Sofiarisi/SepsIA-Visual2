@echo off
ECHO =================================================
ECHO  1. CREANDO ENTORNO VIRTUAL (venv)...
ECHO =================================================
python -m venv venv

ECHO.
ECHO =================================================
ECHO  2. ACTIVANDO ENTORNO VIRTUAL...
ECHO =================================================
CALL venv\Scripts\activate.bat

ECHO.
ECHO =================================================
ECHO  3. INSTALANDO DEPENDENCIAS (requirements.txt)...
ECHO =================================================
pip install -r requirements.txt

ECHO.
ECHO =================================================
ECHO  4. CORRIENDO API (Api.py)...
ECHO =================================================
python Api.py

ECHO.
ECHO La API se ha detenido. Presiona una tecla para salir.
PAUSE