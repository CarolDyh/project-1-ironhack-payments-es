# 0 ##### CARGAR librerias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configuración de estilo
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
# ====================================================================================================================================================================================
#                                                               Análisis de Cohortes para los Pagos de Ironhack (Proyecto 1) 
# ====================================================================================================================================================================================

# 1 ##### CARGAR DATOS
# Cargar archivos Excel descripcion de las columnas de los csv
# lexique_df = pd.read_excel("/Users/tonzo/Desktop/Data_Science_ironhack/project-1-ironhack-payments-es-main/project_dataset/LexiqueDataAnalyst.xlsx")
# Cargar archivos csv
cash_df = pd.read_csv("/Users/tonzo/Desktop/Data_Science_ironhack/project-1-ironhack-payments-es-main/project_dataset/extract_cash_request_dataanalyst.csv")
fees_df = pd.read_csv("/Users/tonzo/Desktop/Data_Science_ironhack/project-1-ironhack-payments-es-main/project_dataset/extract_fees_dataanalyst.csv")

# Ver primeras filas de cada dataset
# print("CASH")
# print(cash_df.head())
# print("\nFEES")
# print(fees_df.head())
# print("\nLEXIQUE")
# print(lexique_df.head())

# ===============================================
# ANALSIS EXPLORATORIO DE DATOS (DEA)
# ===============================================
# 2.1 ## Dimensiones y caracteristicas

# print(cash_df.info()) #16 columnas 23970 registros
# print(cash_df.describe(include='all'))

# print(fees_df.info()) #13 columnas 21061 registros
# print(fees_df.describe(include='all'))

# 2.2 ## Ver columnas únicas y valores nulos

# for name, df in [("cash", cash_df), ("fees", fees_df)]:
#     print(f"\n{name.upper()} - Valores nulos por columna:")
#     print(df.isnull().sum())
#     print(f"{name.upper()} - Valores únicos por columna:")
#     print(df.nunique())

# ----------------------
# ANÁLISIS DE VALORES NULOS
# ----------------------
# missing_fees = fees_df.isnull().mean() * 100
# missing_cash = cash_df.isnull().mean() * 100

# missing_df = pd.DataFrame({
#     'fees (%)': missing_fees,
#     'cash_requests (%)': missing_cash
# }).sort_values(by='fees (%)', ascending=False)

# print("Porcentaje de valores nulos por columna:\n")
# print(missing_df)

#                              fees (%)  cash_requests (%)
# category                    89.573145                NaN
# from_date                   63.126157                NaN
# to_date                     63.126157                NaN
# paid_at                     26.257063                NaN
# cash_request_id              0.018992                NaN
# charge_moment                0.000000                NaN
# created_at                   0.000000           0.000000
# id                           0.000000           0.000000
# reason                       0.000000                NaN
# status                       0.000000           0.000000
# total_amount                 0.000000                NaN
# type                         0.000000                NaN
# updated_at                   0.000000           0.000000
# amount                            NaN           0.000000
# cash_request_received_date        NaN          32.044222
# deleted_account_id                NaN          91.222361
# moderated_at                      NaN          33.103880
# money_back_date                   NaN          30.984564
# reco_creation                     NaN          86.107635
# reco_last_update                  NaN          86.107635
# recovery_status                   NaN          86.107635
# reimbursement_date                NaN           0.000000
# send_at                           NaN          30.575720
# transfer_type                     NaN           0.000000
# user_id                           NaN           8.773467


# ----------------------
# HISTOGRAMAS DE MONTOS
# ----------------------
# sns.histplot(cash_df["amount"], bins=30, kde=True)
# plt.title("Distribución de montos en solicitudes de efectivo")
# plt.xlabel("Monto (€)")
# plt.ylabel("Frecuencia")
# plt.show()

# sns.histplot(fees_df["total_amount"], bins=30, kde=True)
# plt.title("Distribución de montos en fees")
# plt.xlabel("Monto (€)")
# plt.ylabel("Frecuencia")
# plt.show()


# ----------------------
# CONTEO DE STATUS
# ----------------------
# sns.countplot(data=cash_df, x="status", order=cash_df["status"].value_counts().index)
# plt.title("Distribución de status en cash requests")
# plt.xlabel("Status")
# plt.ylabel("Frecuencia")
# plt.xticks(rotation=45)
# plt.show()

# sns.countplot(data=fees_df, x="status", order=fees_df["status"].value_counts().index)
# plt.title("Distribución de status en fees")
# plt.xlabel("Status")
# plt.ylabel("Frecuencia")
# plt.xticks(rotation=45)
# plt.show()

## Normalizamos nombres de columnas
cash_df.columns = cash_df.columns.str.strip().str.lower().str.replace(" ", "_")
# print(cash_df.head(5))
fees_df.columns = fees_df.columns.str.strip().str.lower().str.replace(" ", "_")
# print(fees_df.head(5))

# Convertir fechas
# #Cash
# cash_df['created_at'] = pd.to_datetime(cash_df['created_at'], errors='coerce')
fechas_cash = [
    'created_at', 'updated_at', 'moderated_at', 'reimbursement_date', 'cash_request_received_date',
    'money_back_date', 'send_at', 'reco_creation', 'reco_last_update']

for col in fechas_cash:
    if col in cash_df.columns:
        cash_df[col] = pd.to_datetime(cash_df[col], errors='coerce')

# # #fees
fechas_fees = [
    'created_at', 'updated_at',	'paid_at','from_date','to_date']

for col in fechas_fees:
    if col in fees_df.columns:
        fees_df[col] = pd.to_datetime(fees_df[col], errors='coerce')


# ----------------------
# EVOLUCIÓN TEMPORAL DE SOLICITUDES
# ----------------------
# cash_df["month_created"] = cash_df["created_at"].dt.to_period("M")
# monthly_requests = cash_df["month_created"].value_counts().sort_index()

# monthly_requests.plot(kind='bar')
# plt.title("Solicitudes de efectivo por mes")
# plt.xlabel("Mes")
# plt.ylabel("Cantidad de solicitudes")
# plt.xticks(rotation=90)
# plt.show()


# ----------------------
# ESTADÍSTICAS DESCRIPTIVAS BÁSICAS
# ----------------------
print("\nEstadísticas descriptivas de 'amount' en cash requests:")
print(cash_df["amount"].describe())

print("\nEstadísticas descriptivas de 'total_amount' en fees:")
print(fees_df["total_amount"].describe())

# Estadísticas descriptivas de 'amount' en cash requests:
# count    23970.000000
# mean        82.720818
# std         26.528065
# min          1.000000
# 25%         50.000000
# 50%        100.000000
# 75%        100.000000
# max        200.000000
# Name: amount, dtype: float64

# Estadísticas descriptivas de 'total_amount' en fees:
# count    21061.000000
# mean         5.000237
# std          0.034453
# min          5.000000
# 25%          5.000000
# 50%          5.000000
# 75%          5.000000
# max         10.000000

# ----------------------
# DISTRIBUCIÓN DE TIPO DE FEE
# ----------------------
# sns.countplot(data=fees_df, x="type", order=fees_df["type"].value_counts().index)
# plt.title("Tipos de fees")
# plt.xlabel("Tipo")
# plt.ylabel("Frecuencia")
# plt.xticks(rotation=45)
# plt.show()

# ----------------------
# MIRAMOS DUPLICADOS
# ----------------------
# #Cash
# # print("Duplicados exactos:", cash_df.duplicated().sum())
# # print("Duplicados por id:", cash_df.duplicated(subset='id').sum())

# # # Si hay duplicados por id >>>> NO LOS HAY
# # # cash_df = cash_df.drop_duplicates(subset='id')

# #Fees

# print("Duplicados exactos:", fees_df.duplicated().sum())
# print("Duplicados por id:", fees_df.duplicated(subset='id').sum())

# # Si hay duplicados por id>>>> NO LOS HAY
# # cash_df = fees_df.drop_duplicates(subset='id')

# ----------------------
# Transformacion de variables
# ----------------------

# # CASH ###### Convertimos col a categoricas si hace falta
# cat_cols = ['status', 'transfer_type', 'recovery_status']

# for col in cat_cols:
#     if col in cash_df.columns:
#         cash_df[col] = cash_df[col].astype('category')

# Boxplot para ver valores extremos o outliers
# sns.boxplot(y=cash_df['amount'])
# plt.title('Distribución del monto solicitado (amount) pre outliers proc')
# plt.xticks(rotation=0)
# plt.show()

# #### No eliminaremos registros porque no consideramos outliers los pagos de 1 a 200
# # # Eliminar outliers si es necesario
# # q1 = cash_df['amount'].quantile(0.25)
# # q3 = cash_df['amount'].quantile(0.75)
# # iqr = q3 - q1
# # lower_bound = q1 - 1.5 * iqr
# # upper_bound = q3 + 1.5 * iqr
# #  Filtrar valores válidos
# # # cash_df = cash_df[(cash_df['amount'] >= lower_bound) & (cash_df['amount'] <= upper_bound)]


# # VALORES NULOS
# # print("Nulos en 'user_id':", cash_df['user_id'].isna().sum())
# cash_df['user_final_id'] = cash_df['user_id'].combine_first(cash_df['deleted_account_id'])
# # # print("Nulos en 'user_final_id':", cash_df['user_final_id'].isna().sum()) #ahora hay 0 valores nulos de id
# # print(cash_df.head(20))
# #### La columna moderated_at tiene 7935 valores nulos, que se corresponden a el status "rejected" en la columna status

# ## FEES ###### Convertimos col a categoricas si hace falta
# categoricas_fees = ['type', 'status', 'category', 'charge_moment']

# for col in categoricas_fees:
#     if col in fees_df.columns:
#         fees_df[col] = fees_df[col].astype('category')


# # sns.boxplot(x=fees_df['total_amount'])
# # plt.title('Distribución de los montos de comisiones (total_amount)')
# # plt.show()

# print(fees_df['total_amount'].describe())
# # # Eliminar outlier unico con valor 10 (el resto de registros es de 5 y no tiene logica este valor diferente)
fees_df = fees_df[fees_df['total_amount'] <= 5]
# sns.boxplot(x=fees_df['total_amount'])
# plt.title('Distribución de los montos de comisiones (total_amount)')
# plt.show()

# # print("Valores nulos por columna en fees_df:")
# # print(fees_df.isnull().sum())
# # En el caso de los valores nulos en la columna 'Category', no seran tratados porque tienen relacion con la colmna 'type' (cuando 'type' es 'incident' no hay valores nulos en "category")

# # print("Duplicados exactos:", fees_df.duplicated().sum())
# # print("Duplicados por id:", fees_df.duplicated(subset='id').sum())

# # # # Si hay duplicados por ID (debería ser único)
# # # fees_df = fees_df.drop_duplicates(subset='id')NO HAY VALORES DUPLICADOS
# # print("Nulos en 'charge_moment':", fees_df['charge_moment'].isna().sum())
# # fees_df['control'] = fees_df['from_date'].notna().astype(int)
# # tabla_pivot = (
# #     fees_df
# #     .groupby(['type', 'control'])
# #     .size()
# #     .unstack(fill_value=0)
# # )

# # print(tabla_pivot)
# #Esto es con charge_moment
# # control            0     1
# # charge_moment             
# # after          13295  3429
# # before             0  4337

# #Esto es con type
# # control              0     1
# # type                        
# # incident          2196     0
# # instant_payment  11099     0
# # postpone             0  7766


# ----------------------
# ANALISIS DE DATOS
# ----------------------

# # Obtener el primer cash request por usuario
# cash_df['user_final_id'] = cash_df['user_final_id'].astype('Int64').astype(str).astype('category')
# # print(cash_df.sort_values('user_final_id', ascending = False))

# #Nos quedamos con aquellos registros creados a partir del 2020 (fecha en la que segun el enunciado se inicia esta metodologia), 
# #ya que los datos anteriores podrian haber sido gestionados bajo otros criterios y las metricas podrian verse afectadas.
# cash_df = cash_df[cash_df['created_at'] >= '2020-01-01']

# #Definimos la cohorte en base a la fecha del primer cash request
# first_cash = (
#     cash_df.groupby('user_final_id')['created_at']
#     .min()
#     .dt.to_period('M')
#     .reset_index()
#     .rename(columns={'created_at': 'cohort_month'})
# )
# # print(first_cash)

# # # Agregar la cohorte a cada fila del cash_df
# cash_df = cash_df.merge(first_cash, on='user_final_id', how= 'left')
# print(cash_df[['user_final_id', 'created_at', 'cohort_month']].head())

# # print(cash_df)
# # #4. Crear columna del mes de la transacción
# cash_df['transaction_month'] = cash_df['created_at'].dt.to_period('M')
# # print(cash_df)
# # Calcular meses desde la cohorte
# cash_df['cohort_index'] = (cash_df['transaction_month'] - cash_df['cohort_month']).apply(lambda x: x.n)


# #MÉTRICA 1: Frecuencia de uso del servicio
# frecuencia = (
#     cash_df.groupby(['cohort_month', 'cohort_index'])['id']
#     .count()
#     .unstack(fill_value=0)
# )

# # plt.figure(figsize=(14, 6))
# # sns.heatmap(frecuencia, cmap='Blues', annot=True, fmt='d')
# # plt.title("Frecuencia de uso del servicio por cohorte y mes")
# # plt.ylabel("Cohorte (mes del primer cash)")
# # plt.xlabel("Mes desde el primer cash")
# # plt.tight_layout()
# # plt.show()



# #MÉTRICA 2: Tasa de incidentes por cohorte
# # Asegurar que estén en el mismo tipo
# fees_df['cash_request_id'] = fees_df['cash_request_id'].astype('Int64')

# # # Filtrar incidentes
# # incidentes = fees_df[fees_df['type'] == 'incident']

# # # Marcar incidentes en cash_df
# # cash_df['has_incident'] = cash_df['id'].isin(incidentes['cash_request_id']).astype(int)

# # # Calcular tasa de incidentes por cohorte y mes
# # tasa_incidentes = (
# #     cash_df.groupby(['cohort_month', 'cohort_index'])['has_incident']
# #     .mean()
# #     .unstack(fill_value=0)
# # )

# # # Visualizar
# # plt.figure(figsize=(14, 6))
# # sns.heatmap(tasa_incidentes, cmap='Reds', annot=True, fmt='.2f')
# # plt.title("Tasa de incidentes por cohorte y mes")
# # plt.ylabel("Cohorte")
# # plt.xlabel("Mes desde el primer cash")
# # plt.tight_layout()
# # plt.show()

# ####
# #MÉTRICA 3: Ingresos generados por cohorte
# #Nuestra cohorte esta definida por cohort_month
# merged = fees_df.merge(cash_df, left_on="cash_request_id", right_on="id", suffixes=('_fees', '_cash'))

# #alcular ingresos por cohorte (suma de total_amount):
# # ingresos_por_cohorte = merged.groupby('cohort_month')['total_amount'].sum()
# # print(ingresos_por_cohorte)

# # plt.figure(figsize=(12, 6))
# # ingresos_por_cohorte.plot(kind='bar')
# # plt.title('Ingresos generados por cohorte')
# # plt.xlabel('Cohorte (mes de primera solicitud de efectivo)')
# # plt.ylabel('Ingresos (€)')
# # plt.xticks(rotation=45)
# # plt.tight_layout()
# # plt.show()


# # #MÉTRICA 4: 

# # Filtrar filas donde el tipo de fee es 'incident'
# # incidencias = merged[merged['type'] == 'incident']

# # # Contar incidencias por usuario y cohorte
# # incidencias_por_usuario = incidencias.groupby(['user_id', 'cohort_month']).size().reset_index(name='n_incidencias')

# # # Filtrar usuarios con más de 3 incidencias
# # usuarios_preocupantes = incidencias_por_usuario[incidencias_por_usuario['n_incidencias'] >= 1 ]

# # # Contar usuarios preocupantes por cohorte
# # preocupantes_por_cohorte = usuarios_preocupantes.groupby('cohort_month')['user_id'].nunique().sort_index()

# # # Total de usuarios únicos por cohorte
# # usuarios_totales_por_cohorte = merged.groupby('cohort_month')['user_id'].nunique()

# # # Calcular proporción
# # proporcion_preocupantes = (preocupantes_por_cohorte / usuarios_totales_por_cohorte).dropna()

# # # Visualizar
# # plt.figure(figsize=(12, 6))
# # proporcion_preocupantes.plot(kind='bar', color='steelblue')
# # plt.title('Proporción de usuarios con incidencias por cohorte')
# # plt.xlabel('Cohorte (mes de primera solicitud de efectivo)')
# # plt.ylabel('Proporción de usuarios preocupantes')
# # plt.xticks(rotation=45)
# # plt.tight_layout()
# # plt.show()


# ###Usuarios preocupantes
# # Filtrar filas donde el tipo de fee es 'incident'
# incidencias = merged[merged['type'] == 'incident']

# # Contar incidencias por usuario y cohorte
# incidencias_por_usuario = incidencias.groupby(['user_id', 'cohort_month']).size().reset_index(name='n_incidencias')
# usuarios_preocupantes = incidencias_por_usuario[incidencias_por_usuario['n_incidencias'] >= 3 ]
# print(usuarios_preocupantes)
# # # (Opcional) Guardar en CSV
# usuarios_preocupantes.to_csv("usuarios_mas_de_3_incidencias.csv", index=False)