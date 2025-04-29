import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import config

# Configuración de visualización
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12


def get_unique_original_iterations():
    """
    Extrae los valores únicos de 'original_iteration' de los archivos en MEDICINE_COUNTERFACTUAL_PATH_3

    Returns:
        Lista de valores únicos de original_iteration
    """
    # Obtener la ruta desde config
    counterfactual_dir = config.MEDICINE_COUNTERFACTUAL_PATH_3

    # Encontrar todos los archivos CSV en el directorio de counterfactuals
    csv_files = glob.glob(os.path.join(counterfactual_dir, "*.csv"))
    print(f"Se encontraron {len(csv_files)} archivos CSV en {counterfactual_dir}")

    if not csv_files:
        print("No se encontraron archivos CSV. Por favor verifica la ruta del directorio.")
        return []

    # Lista para almacenar todos los valores de original_iteration
    all_iterations = []

    # Leer cada archivo y extraer valores de original_iteration
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if 'original_iteration' in df.columns:
                iterations = df['original_iteration'].unique()
                all_iterations.extend(iterations)
                print(f"Encontrados {len(iterations)} valores en {os.path.basename(file)}")
            else:
                print(f"Advertencia: {os.path.basename(file)} no contiene la columna 'original_iteration'")
        except Exception as e:
            print(f"Error al leer {os.path.basename(file)}: {str(e)}")

    # Obtener valores únicos
    unique_iterations = sorted(set(all_iterations))
    print(f"Total de valores únicos de original_iteration: {len(unique_iterations)}")
    print(f"Valores únicos: {unique_iterations}")

    return unique_iterations


def visualize_preference_ratios(df, title, output_dir):
    """
    Visualiza las distribuciones de ratios de preferencia para un dataframe dado.

    Args:
        df: DataFrame con los resultados
        title: Título para la visualización
        output_dir: Directorio para guardar las visualizaciones
    """
    print(f"Procesando datos para {title}")
    print(f"Tamaño de los datos: {len(df)} filas")

    # Usamos todos los datos, sin filtrar por threshold de error
    df_good = df.copy()

    print(f"Usando todos los datos: {len(df_good)} filas")

    # Función para extraer columnas de parámetros
    def get_column(df, col_name):
        if col_name in df.columns:
            return df[col_name]

        # Intentar coincidencia parcial
        matches = [col for col in df.columns if col_name in col]
        if matches:
            return df[matches[0]]

        # Si no se encuentra, devolver columna de NaN
        print(f"Advertencia: Columna '{col_name}' no encontrada")
        return pd.Series(np.nan, index=df.index)

    # Extraer columnas relevantes
    df_good['axax'] = get_column(df_good, "(('A', 'X'), ('A', 'X'))")
    df_good['axay'] = get_column(df_good, "(('A', 'X'), ('A', 'Y'))")
    df_good['bxbx'] = get_column(df_good, "(('B', 'X'), ('B', 'X'))")
    df_good['bxby'] = get_column(df_good, "(('B', 'X'), ('B', 'Y'))")
    df_good['axbx'] = get_column(df_good, "(('A', 'X'), ('B', 'X'))")

    # Calcular ratios
    df_good['ratio_axax_axay'] = df_good['axax'] / df_good['axay']
    df_good['ratio_bxbx_bxby'] = df_good['bxbx'] / df_good['bxby']
    df_good['ratio_axax_axbx'] = df_good['axax'] / df_good['axbx']

    # Eliminar infinitos y NaN
    df_good.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_filtered = df_good.dropna(subset=['ratio_axax_axay', 'ratio_bxbx_bxby', 'ratio_axax_axbx'])

    # Filtrar usando el threshold de 10
    df_vis = df_filtered[
        (df_filtered['ratio_axax_axay'] > 0.1) & (df_filtered['ratio_axax_axay'] < 10) &
        (df_filtered['ratio_bxbx_bxby'] > 0.1) & (df_filtered['ratio_bxbx_bxby'] < 10) &
        (df_filtered['ratio_axax_axbx'] > 0.1) & (df_filtered['ratio_axax_axbx'] < 10)
        ]

    print(f"Datos disponibles para visualización: {len(df_vis)}")

    if len(df_vis) == 0:
        print("No hay suficientes datos para visualización.")
        return None

    # Crear visualizaciones
    # 1. Distribuciones de los ratios
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Primera fila: Histogramas con notación LaTeX
    sns.histplot(df_vis['ratio_axax_axay'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title(r'Distribution of ratio $f(A_x,A_x)$ vs $f(A_x,A_y)$')
    axes[0, 0].axvline(x=1, color='r', linestyle='--')
    axes[0, 0].set_xlabel(r'$f(A_x,A_x) \div f(A_x,A_y)$')

    sns.histplot(df_vis['ratio_bxbx_bxby'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title(r'Distribution of ratio $f(B_x,B_x)$ vs $f(B_x,B_y)$')
    axes[0, 1].axvline(x=1, color='r', linestyle='--')
    axes[0, 1].set_xlabel(r'$f(B_x,B_x) \div f(B_x,B_y)$')

    sns.histplot(df_vis['ratio_axax_axbx'], kde=True, ax=axes[0, 2])
    axes[0, 2].set_title(r'Distribution of ratio $f(A_x,A_x)$ vs $f(A_x,B_x)$')
    axes[0, 2].axvline(x=1, color='r', linestyle='--')
    axes[0, 2].set_xlabel(r'$f(A_x,A_x) \div f(A_x,B_x)$')

    # Segunda fila: Diagramas de dispersión de los valores originales con notación LaTeX
    sns.scatterplot(x='axax', y='axay', data=df_vis, ax=axes[1, 0])
    axes[1, 0].plot([0, 1], [0, 1], 'r--')
    axes[1, 0].set_title(r'$f(A_x,A_x)$ vs $f(A_x,A_y)$')
    axes[1, 0].set_xlabel(r'$f(A_x,A_x)$: Preference of A-type individuals from same classroom')
    axes[1, 0].set_ylabel(r'$f(A_x,A_y)$: Preference of A-type individuals from different classrooms')

    sns.scatterplot(x='bxbx', y='bxby', data=df_vis, ax=axes[1, 1])
    axes[1, 1].plot([0, 1], [0, 1], 'r--')
    axes[1, 1].set_title(r'$f(B_x,B_x)$ vs $f(B_x,B_y)$')
    axes[1, 1].set_xlabel(r'$f(B_x,B_x)$: Preference of B-type individuals from same classroom')
    axes[1, 1].set_ylabel(r'$f(B_x,B_y)$: Preference of B-type individuals from different classrooms')

    sns.scatterplot(x='axax', y='axbx', data=df_vis, ax=axes[1, 2])
    axes[1, 2].plot([0, 1], [0, 1], 'r--')
    axes[1, 2].set_title(r'$f(A_x,A_x)$ vs $f(A_x,B_x)$')
    axes[1, 2].set_xlabel(r'$f(A_x,A_x)$: Preference of A-type individuals from same classroom')
    axes[1, 2].set_ylabel(r'$f(A_x,B_x)$: Preference of A-type individuals for B-type individuals')

    # Tercera fila: Boxplots con notación LaTeX
    sns.boxplot(y=df_vis['ratio_axax_axay'], ax=axes[2, 0])
    axes[2, 0].set_title(r'Boxplot of ratio $f(A_x,A_x) \div f(A_x,A_y)$')
    axes[2, 0].axhline(y=1, color='r', linestyle='--')
    axes[2, 0].set_ylabel(r'Ratio $f(A_x,A_x) \div f(A_x,A_y)$')

    sns.boxplot(y=df_vis['ratio_bxbx_bxby'], ax=axes[2, 1])
    axes[2, 1].set_title(r'Boxplot of ratio $f(B_x,B_x) \div f(B_x,B_y)$')
    axes[2, 1].axhline(y=1, color='r', linestyle='--')
    axes[2, 1].set_ylabel(r'Ratio $f(B_x,B_x) \div f(B_x,B_y)$')

    sns.boxplot(y=df_vis['ratio_axax_axbx'], ax=axes[2, 2])
    axes[2, 2].set_title(r'Boxplot of ratio $f(A_x,A_x) \div f(A_x,B_x)$')
    axes[2, 2].axhline(y=1, color='r', linestyle='--')
    axes[2, 2].set_ylabel(r'Ratio $f(A_x,A_x) \div f(A_x,B_x)$')

    plt.tight_layout()
    fig.suptitle(f'Preference Ratio Distributions - {title}', fontsize=16, y=0.99)
    plt.subplots_adjust(top=0.95)

    # Guardar figura
    safe_title = title.replace(' ', '_').replace('/', '_')
    output_path = os.path.join(output_dir, f"preference_ratio_{safe_title}.png")
    plt.savefig(output_path)
    print(f"Visualización guardada en: {output_path}")

    # Mostrar estadísticas con notación LaTeX
    print("\nEstadísticas de los ratios de preferencia:")

    stats_summary = []

    for ratio, name in zip(
            ['ratio_axax_axay', 'ratio_bxbx_bxby', 'ratio_axax_axbx'],
            [r'f(A_x,A_x) ÷ f(A_x,A_y)', r'f(B_x,B_x) ÷ f(B_x,B_y)', r'f(A_x,A_x) ÷ f(A_x,B_x)']
    ):
        values = df_vis[ratio].dropna()
        if len(values) > 0:
            stats = {
                'title': title,
                'ratio': name,
                'mean': values.mean(),
                'median': values.median(),
                'std_dev': values.std(),
                'min': values.min(),
                'max': values.max(),
                'pct_above_1': (values > 1).mean() * 100  # % de valores > 1
            }

            stats_summary.append(stats)

            print(f"\n{name}:")
            print(f"  Media: {stats['mean']:.4f}")
            print(f"  Mediana: {stats['median']:.4f}")
            print(f"  Desv. Estándar: {stats['std_dev']:.4f}")
            print(f"  Rango: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  % valores > 1: {stats['pct_above_1']:.2f}%")

            if ratio == 'ratio_axax_axay':
                if stats['mean'] > 1:
                    interpretacion = "A-type individuals prefer interacting with A-type individuals from same classroom"
                else:
                    interpretacion = "A-type individuals prefer interacting with A-type individuals from different classrooms"
            elif ratio == 'ratio_bxbx_bxby':
                if stats['mean'] > 1:
                    interpretacion = "B-type individuals prefer interacting with B-type individuals from same classroom"
                else:
                    interpretacion = "B-type individuals prefer interacting with B-type individuals from different classrooms"
            else:  # ratio_axax_axbx
                if stats['mean'] > 1:
                    interpretacion = "A-type individuals prefer interacting with A-type vs B-type individuals"
                else:
                    interpretacion = "A-type individuals prefer interacting with B-type vs A-type individuals"

            print(f"  Interpretación: {interpretacion}")

    return stats_summary


def process_economics_by_iterations():
    """
    Procesa el archivo de Economía, filtrando directamente por los valores de iteration
    que coinciden con los original_iteration en los archivos de counterfactual
    """
    # Obtener valores únicos de original_iteration
    unique_iterations = get_unique_original_iterations()

    if not unique_iterations:
        print("No se pudieron encontrar valores de original_iteration. Finalizando.")
        return

    # Cargar el archivo de Economía
    economics_file_path = os.path.join(config.GENERAL_PARAMETER_PATH, "20250312_225502_sa_results_Medicine_201610.csv")
    print(f"\nCargando archivo de Economía: {economics_file_path}")

    try:
        economics_df = pd.read_csv(economics_file_path)
        print(f"Archivo de Economía cargado. Tamaño: {len(economics_df)} filas")
    except Exception as e:
        print(f"Error al cargar el archivo de Economía: {str(e)}")
        return

    # Verificar si 'iteration' está en las columnas
    if 'iteration' not in economics_df.columns:
        print("Error: El archivo de Economía no contiene la columna 'iteration'")
        return

    # Filtrar el dataframe de Economía directamente, conservando sólo las filas cuyo 'iteration'
    # coincide con alguno de los valores de 'original_iteration' encontrados
    filtered_economics_df = economics_df[economics_df['iteration'].isin(unique_iterations)]
    print(f"Después de filtrar por las iteraciones de la lista: {len(filtered_economics_df)} filas")

    # Si no hay datos después del filtrado, terminamos
    if len(filtered_economics_df) == 0:
        print("No hay datos después de filtrar por las iteraciones. Finalizando.")
        return

    # Crear directorio de salida
    output_dir = os.path.join(config.RESULTS_DIR_GEN, "economics_filtered_by_iterations")
    os.makedirs(output_dir, exist_ok=True)

    # Analizar el dataframe filtrado completo
    print(f"\n{'-' * 50}")
    print(f"Analizando datos de Economía filtrados por las iteraciones seleccionadas")

    # Ejecutar análisis para el conjunto filtrado
    iteration_stats = visualize_preference_ratios(
        filtered_economics_df,
        f"Medicine - Filtered by Selected Iterations",
        output_dir
    )

    # También guardar el dataframe filtrado como CSV para análisis adicionales
    filtered_path = os.path.join(output_dir, "medicine_filtered_by_iterations.csv")
    filtered_economics_df.to_csv(filtered_path, index=False)
    print(f"\nDataframe filtrado guardado en: {filtered_path}")


def main():
    """Función principal para ejecutar el análisis"""
    # Procesar el archivo de Economía filtrando por iteraciones
    process_economics_by_iterations()

    print("\nAnálisis completado.")


if __name__ == "__main__":
    main()