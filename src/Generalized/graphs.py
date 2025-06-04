import os
import glob
from xxsubtype import bench

import pandas as pd
import numpy as np
import ast
from scipy import stats

import config
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_total_links(df):
    """
    Calcula la columna total_links basada en la fórmula:
    alone = ((A,1),0) x N_A_1 + ((A,2),0) x N_A_2 + ((B,1),0) x N_B_1 + ((B,2),0) x N_B_2
    total_links = (N - alone) / 2

    Args:
        df: DataFrame con los datos originales

    Returns:
        DataFrame con la nueva columna total_links
    """
    df = df.copy()

    # Calcular alone
    alone = (df["(('A', 1), 0)"] * df['N_A_1'] +
             df["(('A', 2), 0)"] * df['N_A_2'] +
             df["(('B', 1), 0)"] * df['N_B_1'] +
             df["(('B', 2), 0)"] * df['N_B_2'])

    # Calcular total_links
    df['total_links'] = (df['N'] - alone) / 2

    return df


def calculate_distribution_stats_total_links(results_df, group_column='counterfactual', metric='total_links'):
    """
    Calcula estadísticas descriptivas para cada distribución basada en total_links.

    Args:
        results_df: DataFrame con los resultados
        group_column: Columna para agrupar las distribuciones
        metric: Métrica a analizar ('total_links' o 'cross_linkedness')

    Returns:
        DataFrame con estadísticas por grupo
    """

    # Verificar si existe la columna de agrupación
    if group_column not in results_df.columns:
        if 'source_file' in results_df.columns:
            group_column = 'source_file'
            print(f"Advertencia: '{group_column}' no encontrada. Usando 'source_file'.")
        else:
            print("Error: No se encontró columna para agrupar.")
            return None

    # Verificar si existe la métrica
    if metric not in results_df.columns:
        print(f"Error: La columna '{metric}' no existe en el DataFrame.")
        return None

    # Calcular estadísticas por grupo
    stats_list = []

    for group_name in results_df[group_column].unique():
        group_data = results_df[results_df[group_column] == group_name][metric]

        # Traducir nombre del grupo
        translated_name = translate_group_names(group_name)

        # Calcular estadísticas básicas
        mean_val = group_data.mean()
        variance_val = group_data.var()
        std_val = group_data.std()
        median_val = group_data.median()

        # Calcular skewness (asimetría) y kurtosis (curtosis)
        skewness_val = stats.skew(group_data)
        kurtosis_val = stats.kurtosis(group_data)  # Excess kurtosis (normal = 0)

        # Calcular percentiles
        q25 = group_data.quantile(0.25)
        q75 = group_data.quantile(0.75)
        iqr = q75 - q25

        # Rango
        min_val = group_data.min()
        max_val = group_data.max()
        range_val = max_val - min_val

        # Número de observaciones
        n_obs = len(group_data)

        stats_dict = {
            'Grupo': translated_name,
            'Métrica': metric,
            'N_Observaciones': n_obs,
            'Media': mean_val,
            'Mediana': median_val,
            'Desv_Estándar': std_val,
            'Varianza': variance_val,
            'Asimetría': skewness_val,
            'Curtosis': kurtosis_val,
            'Percentil_25': q25,
            'Percentil_75': q75,
            'Rango_Intercuartil': iqr,
            'Mínimo': min_val,
            'Máximo': max_val,
            'Rango_Total': range_val
        }

        stats_list.append(stats_dict)

    stats_df = pd.DataFrame(stats_list)
    return stats_df


def translate_group_names(group_name):
    """
    Traduce los nombres de grupos al español.
    """
    translations = {
        'Factual': 'Factual',
        'Balancing of classrooms': 'Balanceo de muestra',
        'Incremented proportion of Low - Income': 'Aumento de proporción de estrato bajo'
    }

    # Buscar coincidencias parciales
    for eng_name, esp_name in translations.items():
        if eng_name.lower() in str(group_name).lower():
            return esp_name

    # Si no encuentra traducción, devolver el nombre original
    return group_name


def interpret_distribution_stats(stats_df):
    """
    Interpreta las estadísticas de las distribuciones.

    Args:
        stats_df: DataFrame con estadísticas calculadas
    """
    print("\n" + "=" * 80)
    print("INTERPRETACIÓN DE LAS DISTRIBUCIONES")
    print("=" * 80)

    for _, row in stats_df.iterrows():
        print(f"\n📊 GRUPO: {row['Grupo']} - MÉTRICA: {row['Métrica']}")
        print("-" * 50)

        # Interpretación de asimetría
        if row['Asimetría'] > 0.5:
            asimetria_texto = "🔴 Sesgada hacia la DERECHA (cola larga a la derecha)"
        elif row['Asimetría'] < -0.5:
            asimetria_texto = "🔵 Sesgada hacia la IZQUIERDA (cola larga a la izquierda)"
        else:
            asimetria_texto = "🟢 Aproximadamente SIMÉTRICA"

        # Interpretación de curtosis
        if row['Curtosis'] > 1:
            curtosis_texto = "📈 LEPTOCÚRTICA (más puntiaguda que normal, colas pesadas)"
        elif row['Curtosis'] < -1:
            curtosis_texto = "📉 PLATICÚRTICA (más plana que normal, colas ligeras)"
        else:
            curtosis_texto = "📊 MESOCÚRTICA (similar a distribución normal)"

        print(f"• Asimetría: {row['Asimetría']:.3f} → {asimetria_texto}")
        print(f"• Curtosis: {row['Curtosis']:.3f} → {curtosis_texto}")
        print(f"• Media: {row['Media']:.3f} | Mediana: {row['Mediana']:.3f}")

        if row['Media'] > row['Mediana']:
            print("  ↳ Media > Mediana: confirma sesgo hacia la derecha")
        elif row['Media'] < row['Mediana']:
            print("  ↳ Media < Mediana: confirma sesgo hacia la izquierda")
        else:
            print("  ↳ Media ≈ Mediana: distribución balanceada")


def print_summary_statistics(stats_df, metric_name):
    """
    Imprime un resumen claro de las estadísticas descriptivas principales.

    Args:
        stats_df: DataFrame con estadísticas calculadas
        metric_name: Nombre de la métrica para el título
    """
    print("\n" + "🔢" * 80)
    print(f"RESUMEN DE ESTADÍSTICAS DESCRIPTIVAS - {metric_name.upper()}")
    print("🔢" * 80)

    for _, row in stats_df.iterrows():
        print(f"\n📋 GRUPO: {row['Grupo']}")
        print("=" * 60)
        print(f"📊 N° de Observaciones: {int(row['N_Observaciones'])}")
        print(f"📈 Media (μ):           {row['Media']:.4f}")
        print(f"📊 Desviación Estándar: {row['Desv_Estándar']:.4f}")
        print(f"📊 Mediana:            {row['Mediana']:.4f}")
        print(f"📊 Varianza:           {row['Varianza']:.4f}")
        print(f"📊 Mínimo:             {row['Mínimo']:.4f}")
        print(f"📊 Máximo:             {row['Máximo']:.4f}")
        print(f"📊 Rango:              {row['Rango_Total']:.4f}")
        print(f"📊 Q25 (P25):          {row['Percentil_25']:.4f}")
        print(f"📊 Q75 (P75):          {row['Percentil_75']:.4f}")
        print(f"📊 Rango Intercuartil: {row['Rango_Intercuartil']:.4f}")

    # Tabla comparativa de medias y desviaciones estándar
    print(f"\n📊 TABLA COMPARATIVA - {metric_name.upper()}")
    print("=" * 80)
    print(f"{'Grupo':<35} {'Media':<12} {'Desv. Est.':<12} {'N':<8}")
    print("-" * 80)
    for _, row in stats_df.iterrows():
        print(
            f"{row['Grupo']:<35} {row['Media']:<12.4f} {row['Desv_Estándar']:<12.4f} {int(row['N_Observaciones']):<8}")
    print("-" * 80)


def plot_total_links_kde_with_stats(results_df, output_file=None, save_stats=True):
    """
    Crea el gráfico KDE para total_links y calcula estadísticas de las distribuciones.

    Args:
        results_df: DataFrame con los resultados (debe incluir columna total_links)
        output_file: Ruta para guardar el gráfico (opcional)
        save_stats: Si guardar las estadísticas en CSV
    """

    # Verificar que existe la columna total_links
    if 'total_links' not in results_df.columns:
        print("Error: La columna 'total_links' no existe. Asegúrate de ejecutar calculate_total_links() primero.")
        return None

    # Calcular estadísticas
    stats_df = calculate_distribution_stats_total_links(results_df, metric='total_links')

    if stats_df is not None:
        # Mostrar tabla de estadísticas
        print("\n" + "=" * 100)
        print("ESTADÍSTICAS DESCRIPTIVAS DE TOTAL_LINKS")
        print("=" * 100)
        print(stats_df.round(4).to_string(index=False))

        # Mostrar resumen de estadísticas principales
        print_summary_statistics(stats_df, "Total Links")

        # Interpretar estadísticas
        interpret_distribution_stats(stats_df)

        # Guardar estadísticas si se solicita
        if save_stats and output_file:
            stats_file = output_file.replace('.png', '_estadisticas_total_links.csv')
            stats_df.to_csv(stats_file, index=False)
            print(f"\n💾 Estadísticas de total_links guardadas en: {stats_file}")

    # Configurar matplotlib para español
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 12

    # Crear el gráfico KDE
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 8))

    # Determinar columna para hue
    hue_column = None
    if 'counterfactual' in results_df.columns:
        hue_column = 'counterfactual'
    elif 'source_file' in results_df.columns:
        hue_column = 'source_file'
        print("Advertencia: 'counterfactual' no encontrada. Usando source_file para hue.")

    if hue_column:
        # Crear una copia del DataFrame para modificar las etiquetas
        plot_df = results_df.copy()
        plot_df[hue_column] = plot_df[hue_column].apply(translate_group_names)

        categories = plot_df[hue_column].unique()

        # Mantener el color coding original
        palette = {}
        for cat in categories:
            if "Factual" in str(cat):
                palette[cat] = "green"
            elif "Balanceo de muestra" in str(cat) or "Balancing of classrooms" in str(cat):
                palette[cat] = "dodgerblue"
            elif "Aumento de proporción" in str(cat) or "Incremented proportion" in str(cat):
                palette[cat] = "orange"
            else:
                default_colors = sns.color_palette(n_colors=len(categories))
                palette[cat] = default_colors[0]

        ax = sns.kdeplot(
            data=plot_df,
            x='total_links',
            hue=hue_column,
            fill=True,
            common_norm=False,
            alpha=0.7,
            linewidth=2,
            palette=palette
        )
    else:
        ax = sns.kdeplot(
            data=results_df,
            x='total_links',
            fill=True,
            alpha=0.6,
            linewidth=2
        )

    plt.title('Estimación de Densidad Kernel del Número Total de Links', fontsize=16, pad=20)
    plt.xlabel('Número Total de Links', fontsize=14)
    plt.ylabel('Probabilidad', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"\n📈 Gráfico KDE de total_links guardado en: {output_file}")

    plt.tight_layout()
    plt.show()

    return stats_df


def calculate_distribution_stats(results_df, group_column='counterfactual'):
    """
    Calcula estadísticas descriptivas para cada distribución basada en cross_linkedness.
    (Función original mantenida para compatibilidad)
    """
    return calculate_distribution_stats_total_links(results_df, group_column, 'cross_linkedness')


def plot_cross_linkedness_kde_with_stats(results_df, output_file=None, benchmark_value=1, save_stats=True):
    """
    Crea el gráfico KDE y calcula estadísticas de las distribuciones.
    Versión corregida que preserva el color coding original.
    """

    # Calcular estadísticas
    stats_df = calculate_distribution_stats(results_df)

    if stats_df is not None:
        # Mostrar tabla de estadísticas
        print("\n" + "=" * 100)
        print("ESTADÍSTICAS DESCRIPTIVAS DE LAS DISTRIBUCIONES")
        print("=" * 100)
        print(stats_df.round(4).to_string(index=False))

        # Mostrar resumen de estadísticas principales
        print_summary_statistics(stats_df, "Cross-Linkedness")

        # Interpretar estadísticas
        interpret_distribution_stats(stats_df)

        # Guardar estadísticas si se solicita
        if save_stats and output_file:
            stats_file = output_file.replace('.png', '_estadisticas.csv')
            stats_df.to_csv(stats_file, index=False)
            print(f"\n💾 Estadísticas guardadas en: {stats_file}")

    # Configurar matplotlib para español
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 12

    # Crear el gráfico KDE original
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 8))

    # Determinar columna para hue
    hue_column = None
    if 'counterfactual' in results_df.columns:
        hue_column = 'counterfactual'
    elif 'source_file' in results_df.columns:
        hue_column = 'source_file'
        print("Advertencia: 'counterfactual' no encontrada. Usando source_file para hue.")

    if hue_column:
        # Crear una copia del DataFrame para modificar las etiquetas
        plot_df = results_df.copy()
        plot_df[hue_column] = plot_df[hue_column].apply(translate_group_names)

        categories = plot_df[hue_column].unique()

        # CORRECCIÓN: Preservar el color coding original
        palette = {}
        for cat in categories:
            if "Factual" in str(cat):
                palette[cat] = "green"  # Verde para Factual (como en original)
            elif "Balanceo de muestra" in str(cat) or "Balancing of classrooms" in str(cat):
                palette[cat] = "dodgerblue"  # Azul para Balancing
            elif "Aumento de proporción" in str(cat) or "Incremented proportion" in str(cat):
                palette[cat] = "orange"  # NARANJA para Incremented proportion (como en original)
            else:
                # Color por defecto si no coincide con ninguna categoría
                default_colors = sns.color_palette(n_colors=len(categories))
                palette[cat] = default_colors[0]

        ax = sns.kdeplot(
            data=plot_df,
            x='cross_linkedness',
            hue=hue_column,
            fill=True,
            common_norm=False,
            alpha=0.7,
            linewidth=2,
            palette=palette
        )
    else:
        ax = sns.kdeplot(
            data=results_df,
            x='cross_linkedness',
            fill=True,
            alpha=0.6,
            linewidth=2
        )

    plt.title('Estimación de Densidad Kernel de Valores de Interconexión Cruzada', fontsize=16, pad=20)
    plt.xlabel('Valor de Interconexión Cruzada', fontsize=14)
    plt.ylabel('Densidad', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    benchmark_line = plt.axvline(x=benchmark_value, color='red', linestyle='--', alpha=0.7,
                                 label='Referencia Aleatoria')

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"\n📈 Gráfico KDE guardado en: {output_file}")

    plt.tight_layout()
    plt.show()

    return stats_df


if __name__ == "__main__":

    type_shares = config.TYPE_SHARES_FOLDER_PATH_GEN
    results_path = config.RESULTS_DIR_GEN

    csv_path = os.path.join(type_shares, "Observed_type_shares_non_zeros_generalized.csv")
    file_pattern = os.path.join(results_path, "*economics*_3.csv")
    files = glob.glob(file_pattern)

    # Procesar cada archivo
    filtered_dfs = []

    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}")

        df = pd.read_csv(file_path)
        filtered_dfs.append(df)
        print(f"  Selected {len(df)} rows from {file_name}")

    # Combinar dataframes
    combined_df = pd.concat(filtered_dfs, ignore_index=True)

    # NUEVO: Calcular total_links
    print("\n🔄 Calculando total_links...")
    combined_df = calculate_total_links(combined_df)
    print(
        f"✅ Columna total_links calculada. Rango: {combined_df['total_links'].min():.2f} - {combined_df['total_links'].max():.2f}")

    economics_cs = 0.21818181818181812
    medicine_cs = 0.32627765064836

    # Gráfico original de cross_linkedness
    output_file_cross = os.path.join(results_path, "kdensity_economics_3.png")
    print("\n📊 Generando gráfico de cross_linkedness...")
    distribution_stats_cross = plot_cross_linkedness_kde_with_stats(
        combined_df,
        output_file=output_file_cross,
        benchmark_value=medicine_cs,
        save_stats=True
    )

    # NUEVO: Gráfico de total_links
    output_file_links = os.path.join(results_path, "kdensity_total_links_economics_3.png")
    print("\n📊 Generando gráfico de total_links...")
    distribution_stats_links = plot_total_links_kde_with_stats(
        combined_df,
        output_file=output_file_links,
        save_stats=True
    )