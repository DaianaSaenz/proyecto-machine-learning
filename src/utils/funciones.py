import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def codificarVariablesDataFrame(df):
    x = df.drop("permanencia", axis=1)
    y = df["permanencia"]

    # codificar variables categóricas
    variables_categoricas = ["genero", "nivel_educativo", "estado_civil", "banda_salarial", "categoria_tarjeta"]
    codificacion = OneHotEncoder(handle_unknown='ignore')
    x_codificadas = pd.DataFrame(codificacion.fit_transform(x[variables_categoricas]).toarray(), columns=codificacion.get_feature_names(variables_categoricas))
    x = x.drop(variables_categoricas, axis=1)
    x = pd.concat([x, x_codificadas], axis=1)
    return {x, y}