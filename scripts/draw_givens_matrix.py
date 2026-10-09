"""Draw the standard rotation used in models.py without external artwork."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

figure, axis = plt.subplots(figsize=(8, 5))
axis.set_xlim(-0.7, 4.5)
axis.set_ylim(-1.2, 4.5)
axis.axis('off')
entries = [['1', '0', '0', '0'], ['0', r'\cos\theta', r'-\sin\theta', '0'],
           ['0', r'\sin\theta', r'\cos\theta', '0'], ['0', '0', '0', '1']]
axis.add_patch(Rectangle((0.5, 0.5), 2, 2, color='#e2edf9'))
for row, entries_row in enumerate(entries):
    for column, entry in enumerate(entries_row):
        axis.text(column, 3-row, '$'+entry+'$', ha='center', va='center', fontsize=25)
axis.plot([-0.45, -0.6, -0.6, -0.45], [3.55, 3.55, -0.55, -0.55], color='#18324a', lw=2)
axis.plot([3.45, 3.6, 3.6, 3.45], [3.55, 3.55, -0.55, -0.55], color='#18324a', lw=2)
axis.text(1.5, 4.05, 'Givens rotation in the (1, 2) plane', ha='center', fontsize=16)
axis.text(1.5, -1.05, 'Zero-based indices; all other coordinates remain unchanged', ha='center', fontsize=11)
figure.savefig(Path(__file__).resolve().parents[1] / 'images/givens-rotation-matrix-construction.png', dpi=180, bbox_inches='tight')
