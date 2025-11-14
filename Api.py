



from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import uvicorn
import xgboost as xgb


modelo = xgb.XGBClassifier()
modelo.load_model("model_completo.json")  # or .bin, .txt, etc.
print("Modelo cargado correctamente")
# modelos = {
#     7: xgb.XGBClassifier(),
#     15: xgb.XGBClassifier(),
#     25: xgb.XGBClassifier(),
#     40: xgb.XGBClassifier()
# }


# modelos[7].load_model("model_7.json")
# modelos[15].load_model("model_15.json")
# modelos[25].load_model("model_25.json")
# modelos[40].load_model("model_40.json")
# #CARGAR TODOS LOS MODELOS
# df = pd.read_csv("dataset.csv")  # ⚠️ Asegurate de tener el archivo
# promedios = df.mean()
# Crear app
app = FastAPI()


# Cargar modelo entrenado




# Definir el esquema de datos que vas a recibir (ejemplo con 4 signos vitales)
#CARGAR TODOS LOS POSIBLES DATOS
# class DatosEntrada(BaseModel):
#    from pydantic import BaseModel


class SepsisInput(BaseModel):
    #  Vital signs (1–8)
    HR: float | None = None                      # Frecuencia cardíaca: aumenta con fiebre o shock séptico.
    O2Sat: float | None = None                   # Saturación de oxígeno: baja indica hipoxia, común en sepsis grave.
    Temp: float | None = None                    # Temperatura corporal: fiebre o hipotermia son signos clásicos.
    SBP: float | None = None                     # Presión sistólica: suele bajar en shock séptico.
    MAP: float | None = None                     # Presión arterial media: refleja la perfusión de órganos.
    DBP: float | None = None                     # Presión diastólica: baja en vasodilatación séptica.
    Resp: float | None = None                    # Frecuencia respiratoria: aumenta por acidosis metabólica.
    EtCO2: float | None = None                   # CO₂ espirado: niveles bajos pueden indicar hipoperfusión.

    #  #
    BaseExcess: float | None = None              # Exceso de base: evalúa equilibrio ácido-base.
    HCO3: float | None = None                    # Bicarbonato: bajo en acidosis metabólica por sepsis.
    FiO2: float | None = None                    # Fracción de oxígeno inspirado: indica necesidad de soporte respiratorio.
    pH: float | None = None                      # Nivel de pH sanguíneo: acidosis sugiere sepsis grave.
    PaCO2: float | None = None                   # Presión parcial de CO₂: refleja respiración y metabolismo.
    SaO2: float | None = None                    # Saturación arterial de oxígeno: baja en hipoxemia séptica.
    AST: float | None = None                     # Enzima hepática: aumenta si hay daño hepático por sepsis.
    #15
    BUN: float | None = None                     # Urea: alto indica fallo renal por hipoperfusión.
    Alkalinephos: float | None = None            # Fosfatasa alcalina: se eleva por daño hepático o biliar.
    Calcium: float | None = None                 # Calcio: puede bajar en sepsis por inflamación sistémica.
    Chloride: float | None = None                # Cloro: desequilibrios reflejan alteraciones metabólicas.
    Creatinine: float | None = None              # Creatinina: alta indica daño renal, común en sepsis.
    Bilirubin_direct: float | None = None        # Bilirrubina directa: aumento por disfunción hepática.
    Glucose: float | None = None                 # Glucosa: puede aumentar por estrés o bajar en sepsis avanzada.
    Lactate: float | None = None                 # Lactato: elevado indica hipoxia tisular, marcador clave de sepsis.
    Magnesium: float | None = None               # Magnesio: bajo puede agravar arritmias en sepsis.
    Phosphate: float | None = None               # Fosfato: alterado en disfunción metabólica.
    #25
    Potassium: float | None = None               # Potasio: cambios reflejan alteraciones renales o acidosis.
    Bilirubin_total: float | None = None         # Bilirrubina total: indica daño hepático o colestasis.
    TroponinI: float | None = None               # Troponina I: alta indica daño cardíaco por shock séptico.
    Hct: float | None = None                     # Hematocrito: baja por dilución o hemorragia.
    Hgb: float | None = None                     # Hemoglobina: mide capacidad de transporte de oxígeno.
    PTT: float | None = None                     # Tiempo de tromboplastina parcial: alterado en coagulopatía séptica.
    WBC: float | None = None                     # Glóbulos blancos: elevados o bajos, ambos posibles en sepsis.
    Fibrinogen: float | None = None              # Fibrinógeno: puede bajar por coagulación intravascular diseminada.
    Platelets: float | None = None               # Plaquetas: bajas en sepsis severa por consumo o destrucción.
    #34
    #  Demographics (35–40)
    Age: float | None = None                     # Edad: mayores tienen mayor riesgo y peor pronóstico.
    Gender: float | None = None                  # Género: algunas diferencias inmunológicas pueden influir.
    Unit1: float | None = None                   # Identificador UCI (MICU): tipo de unidad puede reflejar gravedad.
    Unit2: float | None = None                   # Identificador UCI (SICU): quirúrgica o médica.
    HospAdmTime: float | None = None 
    ICULOS: float | None = None             # Tiempo desde admisión hospitalaria: útil para contexto clínico.
    # ICULOS: float | promedios                 # Horas en UCI: refleja evolución del paciente.

#40
# Endpoint de analisis
@app.post("/analizar")


def analizar(datos: SepsisInput, promedios, modelo):
    # Convertimos los datos de entrada a un array en el mismo orden que las columnas del dataset original
    entrada = np.array([[
        datos.HR,
        datos.O2Sat,
        datos.Temp,
        datos.SBP,
        datos.MAP,
        datos.DBP,
        datos.Resp,
        datos.EtCO2,
        datos.BaseExcess,
        datos.HCO3,
        datos.FiO2,
        datos.pH,
        datos.PaCO2,
        datos.SaO2,
        datos.AST,
        datos.BUN,
        datos.Alkalinephos,
        datos.Calcium,
        datos.Chloride,
        datos.Creatinine,
        datos.Bilirubin_direct,
        datos.Glucose,
        datos.Lactate,
        datos.Magnesium,
        datos.Phosphate,
        datos.Potassium,
        datos.Bilirubin_total,
        datos.TroponinI,
        datos.Hct,
        datos.Hgb,
        datos.PTT,
        datos.WBC,
        datos.Fibrinogen,
        datos.Platelets,
        datos.Age,
        datos.Gender,
        datos.Unit1,
        datos.Unit2,
        datos.HospAdmTime,
        datos.ICULOS
    ]], dtype=float)


    print("Los datos entraron")

#     # CARGAR COLUMNAS DEL DATSET A MANO
#     #PARA QUE ESTO FUNCIONE EL FRONT TENDRIA QUE PASAR TODOS LOS DATOS QUE TIENNE Y NO LOS QUE NO, NO VARIABLE CON NONE
#     promedios = df.mean()
#     valores_recibidos = [v for v in entrada if v is not None]
#     n = len(entrada)
#     posibles = [7, 15, 25, 40]
#     # if n not in posibles:
#     #     for limite in posibles:
#     #         if n < 7:
#     #             raise ValueError("Variables menos de 7")
#     #             #Chequear
#     #         elif n <= limite:
              
#     #             n = limite
#     #             break
#     if n < 7:
#         raise ValueError("Variables menos de 7")
#     elif n < 15:
#         n = 15
#     elif n < 25:
#         n = 25
#     elif n < 40:
#         n=40
#     else
#     raise ValueError("Variables menos de 7")


   

           

#     modelo = modelos[n]

# #lista de los valores que tengo en el bloque, lista de que valores hay por bloque y 
#     columnas = df.columns[:n]
#     valores_completos = []
#     for i, col in enumerate(columnas):
#         if i < len(entrada) and not np.isnan(entrada[i]):
#             val = entrada[i]
#         else:
#             val = promedios[col]
#         valores_completos.append(val)
   


    try:
        prediccion = modelo.predict_proba(entrada)
        print("el resultado es ", prediccion[0])
        return {"resultado": float(prediccion[0][1])}
    except Exception as e:
        print("Error en la predicción:", e)
        return {"error": str(e)}


    # prediccion = modelo.predict(entrada)
    # print("el resultad es ", prediccion[0])
    # return {"resultado": prediccion[0]}
uvicorn.run(app, host="0.0.0.0")

