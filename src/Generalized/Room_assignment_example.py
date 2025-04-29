import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import config
from Room_assignment import Assignment


def main_auxiliar():
    """
    Función auxiliar que analiza específicamente el archivo de exposure_201610_Matemáticas.csv
    con diferentes tamaños de bloques (2, 4, 6, 8, 10) y visualiza los resultados comparativos.
    """
    # Ruta al archivo específico
    file_path = os.path.join(config.EXPOSURES_PATH, "exposure_201610_Matemáticas.csv")

    if not os.path.exists(file_path):
        print(f"El archivo {file_path} no existe.")
        return

    # Crear subdirectorio Example para guardar los resultados
    example_dir = os.path.join(config.ASSIGNMENT_PATH, "Example")
    os.makedirs(example_dir, exist_ok=True)
    print(f"Creado directorio para ejemplos: {example_dir}")

    # Cargar la matriz de exposición original
    original_matrix = pd.read_csv(file_path, index_col=0)

    # Definir los tamaños de bloque a probar
    block_sizes = [2, 4, 6, 8, 10]

    # Almacenar resultados para cada tamaño de bloque
    results = {}

    for num_blocks in block_sizes:
        print(f"\nAplicando modelo con {num_blocks} bloques...")

        # Usamos n_init mayor para más intentos de optimización
        best_result = None
        best_likelihood = float('-inf')

        # Más iteraciones para k=10 para encontrar el mejor modelo
        num_attempts = 15 if num_blocks == 10 else 5

        # Realizamos múltiples intentos para cada tamaño de bloque para evitar mínimos locales
        for attempt in range(num_attempts):
            assignment = Assignment(num_blocks=num_blocks)
            result = assignment.fit_weighted_sbm(original_matrix)

            if result['likelihood'] > best_likelihood:
                best_likelihood = result['likelihood']
                best_result = result
                print(f"  Intento {attempt + 1}: Nuevo mejor likelihood: {best_likelihood:.4f}")

        result = best_result
        print(f"Mejor likelihood para {num_blocks} bloques: {best_likelihood:.4f}")

        # Reconstruir la matriz de exposición estimada desde el modelo
        block_assignments = result['block_assignments']
        block_matrix = result['block_matrix']

        # Crear matriz estimada con las mismas dimensiones que la original
        # Usamos float64 explícitamente para evitar advertencias de tipo de datos
        estimated_matrix = pd.DataFrame(0.0, index=original_matrix.index, columns=original_matrix.columns, dtype=float)

        n = len(original_matrix)
        for i in range(n):
            for j in range(n):
                if i != j:  # Mantener diagonal en 0
                    bi, bj = block_assignments[i], block_assignments[j]
                    estimated_matrix.iloc[i, j] = block_matrix[bi, bj]

        # Calcular el error cuadrático medio para medir la precisión
        mse = ((original_matrix - estimated_matrix) ** 2).mean().mean()
        print(f"Error cuadrático medio para {num_blocks} bloques: {mse:.4f}")

        # Guardar resultados
        results[num_blocks] = {
            'block_assignments': block_assignments,
            'block_matrix': block_matrix,
            'estimated_matrix': estimated_matrix,
            'likelihood': result['likelihood'],
            'mse': mse
        }

        # Guardar las asignaciones en el subdirectorio Example
        assignments_file = os.path.join(example_dir, f"room_assignments_201610_Matemáticas_{num_blocks}.csv")
        assignment_df = pd.DataFrame({
            'Identifier': original_matrix.index,
            'Room': block_assignments
        })
        assignment_df.to_csv(assignments_file, index=False)
        print(f"Block assignments saved to {assignments_file}")

        # Guardar la matriz estimada
        estimated_file = os.path.join(example_dir, f"matrix_estimated_{num_blocks}_blocks.csv")
        estimated_matrix.to_csv(estimated_file)
        print(f"Estimated matrix saved to {estimated_file}")

        # Guardar en el resumen general
        assignment.save_summary(201610, f"Matemáticas_{num_blocks}_bloques", result['likelihood'])

    # Crear visualizaciones comparativas
    visualize_comparisons_simplified(original_matrix, results, example_dir)

    return results


def visualize_comparisons_simplified(original_matrix, results, output_dir):
    """
    Simplified visualizations comparing the original exposure matrix with estimated matrices
    for different block sizes, focusing only on visual representation without text analysis.
    """
    # Calculate MSE for each result if it doesn't exist
    for num_blocks, result in results.items():
        if 'mse' not in result:
            estimated_matrix = result['estimated_matrix']
            mse = ((original_matrix - estimated_matrix) ** 2).mean().mean()
            result['mse'] = mse
            print(f"MSE calculated for {num_blocks} blocks: {mse:.4f}")

    # 1. IMPROVED HEATMAPS
    # Subplot configuration
    fig, axes = plt.subplots(2, 3, figsize=(22, 14), dpi=100)
    axes = axes.flatten()

    # Common configuration for all heatmaps
    vmin = 0
    vmax = original_matrix.max().max()

    # Use a different color palette - blue to orange
    cmap = sns.diverging_palette(220, 20, as_cmap=True)  # Blue to orange

    # Global style adjustment
    plt.style.use('seaborn-v0_8-white')

    # Visualize original matrix with improved style
    ax0 = sns.heatmap(
        original_matrix,
        ax=axes[0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Exposure'}
    )
    axes[0].set_title("Original Exposure Matrix", fontsize=16, pad=20)

    # Improve appearance of labels
    axes[0].tick_params(axis='both', labelsize=9, rotation=45)

    # Visualize estimated matrices for each block size
    sorted_blocks = sorted(results.keys())
    for i, num_blocks in enumerate(sorted_blocks, 1):
        if i < len(axes):
            result = results[num_blocks]
            estimated_matrix = result['estimated_matrix']
            mse = result['mse']
            likelihood = result['likelihood']

            sns.heatmap(
                estimated_matrix,
                ax=axes[i],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={'label': 'Estimated Exposure'}
            )

            # Simple title with key metrics
            axes[i].set_title(
                f"Estimated Matrix ({num_blocks} blocks)\n"
                f"Likelihood: {likelihood:.2f}, MSE: {mse:.4f}",
                fontsize=14,
                pad=20
            )

            # Improve appearance of labels
            axes[i].tick_params(axis='both', labelsize=9, rotation=45)

    # Hide unused axes
    for i in range(len(sorted_blocks) + 1, len(axes)):
        axes[i].axis('off')

    # Global title
    fig.suptitle("Room Assignments with Increasing Number of Blocks",
                 fontsize=22, y=0.98, fontweight='bold')

    # Adjust spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "comparison_heatmaps_improved.png"), dpi=300, bbox_inches='tight')

    # 2. METRICS GRAPHS
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=100, sharex=True)

    # Prepare data
    blocks = sorted(results.keys())
    likelihoods = [results[num]['likelihood'] for num in blocks]
    mse_values = [results[num]['mse'] for num in blocks]

    # Subplot 1: Likelihood
    color = '#3498db'  # Blue
    ax1.plot(blocks, likelihoods, 'o-', color=color, linewidth=3, markersize=10)
    ax1.set_ylabel('Log-Likelihood', fontsize=14, color=color)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_title('Likelihood by Number of Blocks', fontsize=16, pad=20)

    # Add values above points
    for i, v in enumerate(likelihoods):
        ax1.text(blocks[i], v + 0.1, f'{v:.2f}', color=color, fontweight='bold',
                 ha='center', va='bottom', fontsize=10)

    # Subplot 2: MSE
    color = '#e74c3c'  # Red
    ax2.plot(blocks, mse_values, 'o-', color=color, linewidth=3, markersize=10)
    ax2.set_xlabel('Number of Blocks', fontsize=14)
    ax2.set_ylabel('Mean Squared Error (MSE)', fontsize=14, color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title('Mean Squared Error by Number of Blocks', fontsize=16, pad=20)

    # Add values above points
    for i, v in enumerate(mse_values):
        ax2.text(blocks[i], v + 0.0002, f'{v:.4f}', color=color, fontweight='bold',
                 ha='center', va='bottom', fontsize=10)

    # Configure shared x axis
    ax2.set_xticks(blocks)
    ax2.set_xticklabels(blocks, fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison_improved.png"), dpi=300, bbox_inches='tight')

    # 3. BAR CHART FOR MSE
    plt.figure(figsize=(10, 6))

    # Create bar chart for MSE with improved style
    bars = plt.bar(blocks, mse_values, color='#3498db', alpha=0.8, width=0.6)

    # Add value labels above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.0001,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add titles and labels
    plt.title('Mean Squared Error by Number of Blocks', fontsize=16, pad=20)
    plt.xlabel('Number of Blocks', fontsize=14)
    plt.ylabel('Mean Squared Error', fontsize=14)
    plt.xticks(blocks, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Improve design
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mse_comparison_bars.png"), dpi=300, bbox_inches='tight')

    # Save raw data for reference
    metrics_df = pd.DataFrame({
        'num_blocks': sorted(blocks),
        'likelihood': likelihoods,
        'mse': mse_values
    })
    metrics_df.to_csv(os.path.join(output_dir, "block_metrics.csv"), index=False)


if __name__ == "__main__":
    # Esto permite ejecutar el análisis directamente si se ejecuta este archivo
    results = main_auxiliar()
    print("Análisis completado.")