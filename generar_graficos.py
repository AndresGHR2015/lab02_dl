import matplotlib.pyplot as plt
import numpy as np
import os

# Crear carpeta para guardar las imágenes si no existe
os.makedirs('images', exist_ok=True)

# Tus datos exactos
targets = ['GDS', 'GDS_R1', 'GDS_R2', 'GDS_R3', 'GDS_R4', 'GDS_R5']
f1_micro = [0.5258, 0.7899, 0.7070, 0.8293, 0.6376, 0.6622]
f1_macro = [0.4012, 0.6120, 0.6648, 0.7394, 0.5559, 0.5960]
exact_match = [0.0643, 0.7480, 0.5720, 0.8249, 0.5371, 0.4075]

# --- Gráfico 1: Comparación de F1-Scores ---
x = np.arange(len(targets))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, f1_micro, width, label='F1 Micro', color='#1f77b4')
rects2 = ax.bar(x + width/2, f1_macro, width, label='F1 Macro', color='#ff7f0e')

ax.set_ylabel('Puntuación')
ax.set_title('Comparación de F1-Score (Micro vs Macro) por Variable Objetivo')
ax.set_xticks(x)
ax.set_xticklabels(targets)
ax.legend(loc='lower right')
ax.set_ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('images/f1_scores.png', dpi=300)
plt.close()

# --- Gráfico 2: Exact Match ---
fig, ax = plt.subplots(figsize=(8, 5))
rects3 = ax.bar(x, exact_match, width=0.5, color='#2ca02c', edgecolor='black')

ax.set_ylabel('Puntuación Exact Match')
ax.set_title('Rendimiento de Coincidencia Exacta (Exact Match)')
ax.set_xticks(x)
ax.set_xticklabels(targets)
ax.set_ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('images/exact_match.png', dpi=300)
plt.close()

