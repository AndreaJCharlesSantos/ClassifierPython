import pandas as pd

def main():

    #Procesamiento del csv
    data = pd.read_csv('wine.csv', delimiter=',')
    #Data original
    print(data)

    #Cambio del nombre de columnas
    data.columns = ["'Alcohol'","'Malic acid'", "'Ash'", "'Alcalinity of ash'", "'Magnesium'", "'Total phenols'", "'Flavanoids'","'Nonflavanoid phenols'","'Proanthocyanins'","'Color intensity'","'Hue'","'OD280/OD315 of diluted wines'","'Proline'", "'class'"]
    
    #Obteniendo orden de columnas
    orden = data.columns.to_list()[0:]
    
    #Preparacion del archivo de salida
    atributos = {"@relation": "'wine'"}

    for columna in orden[:-1]:
        atributos[f"@attribute {columna}"] = f"{'numeric'} "
    atributos["@attribute"] = "class {1.0, 2.0, 3.0}"
    #Creacion del archivo de salida
    with open("wine.arff", "w") as f:
        for key, value in atributos.items():
            f.write(f"{key} {value}\n")
        f.write("@data\n")
        for i, row in data.iterrows():
            f.write(f"{','.join(row.astype(str))}\n")
main()
print("Creado")