# Clasificación de daños en edificios con imágenes satelitales (xBD)

> Modelo de deep learning que clasifica el nivel de daño de los edificios tras desastres naturales, usando imágenes satelitales de antes y después del evento.
> Proyecto en equipo · Grado en Ingeniería Informática · Universidad Carlos III de Madrid (2025–2026)

**Idioma:** [English](./README.md) · Español

[![Python](https://img.shields.io/badge/Python-3-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-torchvision-orange.svg)](https://pytorch.org/)

---

## El problema

Tras un desastre natural, los equipos de rescate necesitan saber rápidamente qué edificios han sufrido daños. La evaluación manual es lenta y peligrosa. El [dataset xBD](https://arxiv.org/abs/1911.09296) ofrece pares de imágenes satelitales de alta resolución (antes y después de 19 desastres, entre ellos terremotos, huracanes, inundaciones, tsunamis e incendios), con cada edificio etiquetado en una escala de cuatro niveles.

La tarea: dado un parche de 64×64 centrado en un edificio, predecir su nivel de daño. Las clases están muy desbalanceadas, y esa es la principal dificultad.

| Clase | Edificios en entrenamiento | Proporción |
| :--- | ---: | ---: |
| Sin daño | 50.928 | 83,3 % |
| Daño menor | 4.659 | 7,6 % |
| Daño mayor | 2.357 | 3,9 % |
| Destruido | 3.227 | 5,3 % |
| **Total** | **61.171** | |

Validación: 8.495 edificios · Test: 22.222 edificios (etiquetas ocultas, evaluado en la competición de la asignatura).

---

## Enfoque

**Fusión temprana de las imágenes de antes y después.** Los parches previo y posterior al desastre se apilan en una única entrada de 6 canales, para que la red compare los dos estados del mismo edificio en lugar de juzgar solo la imagen posterior.

**Dos modelos**

- **CNN propia**, entrenada desde cero: cuatro bloques Conv–BatchNorm–ReLU–MaxPool (de 32 a 256 filtros) seguidos de global average pooling y un clasificador lineal.
- **ResNet-18 preentrenada en ImageNet**, con fine-tuning. Como la entrada tiene 6 canales en lugar de 3, se reconstruyó la primera convolución copiando los pesos RGB preentrenados tanto en los canales de "antes" como en los de "después", conservando las características visuales aprendidas para cada imagen. Se añadió un dropout (p = 0,4) antes del clasificador.

**Data augmentation emparejada** (solo en entrenamiento)

- Transformaciones geométricas, aplicadas **igual** a las dos imágenes: volteos horizontales y verticales, y rotaciones de 0/90/180/270°, ya que las imágenes satelitales no tienen una orientación canónica.
- Transformaciones fotométricas, aplicadas **de forma independiente** a cada imagen: brillo y contraste, porque la iluminación y las condiciones de captura cambian entre fechas.

**Desbalanceo de clases.** Entropía cruzada ponderada por la frecuencia inversa de cada clase. Los modelos se seleccionan por **macro-F1** en validación, que pondera por igual las cuatro clases en lugar de premiar a la mayoritaria.

**Eficiencia.** El pipeline original releía las imágenes TIFF de 1024×1024 para cada parche en cada época. Ahora los parches se preprocesan una sola vez y se guardan en caché como tensores, así que cada época solo lee de memoria. Cada modelo se entrena en unos 3 minutos con GPU.

**Configuración de entrenamiento.** SGD con momento 0,9, batch size 512, 25 épocas y learning rate dividido entre 10 cada 7 épocas (inicial de 1e-2 para la CNN propia; 1e-3 con weight decay de 1e-4 para ResNet-18). Semillas fijas (42) para la reproducibilidad.

---

## Resultados

Conjunto de validación (8.495 edificios). El recall por clase corresponde a la mejor época de cada modelo.

| Modelo | Macro-F1 (mejor época) | Macro-F1 (media de las 10 últimas épocas) | Recall sin daño | Recall daño menor | Recall daño mayor | Recall destruido |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| CNN propia | 0,450 (época 2) | 0,41 | 78 % | 48 % | 0,3 % | 69 % |
| ResNet-18 (fine-tuning, 6 canales) | 0,438 (época 4) | 0,42 | 81 % | 28 % | 6,2 % | 63 % |

**Conjunto de test oculto** (competición de la asignatura en Codabench, 22.222 edificios): el equipo quedó **5.º de la clase**.

| Modelo | Macro-F1 (test) |
| :--- | :---: |
| CNN propia | 0,31 |
| ResNet-18 (fine-tuning, 6 canales) | **0,33** |

<table>
<tr>
<td><img src="figures/confusion_matrix_custom_cnn.png" alt="Matriz de confusión, CNN propia"></td>
<td><img src="figures/confusion_matrix_resnet18.png" alt="Matriz de confusión, ResNet-18"></td>
</tr>
<tr>
<td align="center">CNN propia (mejor época)</td>
<td align="center">ResNet-18 (mejor época)</td>
</tr>
</table>

![Curvas de entrenamiento, ResNet-18](figures/training_curves_resnet18.png)

**Observaciones**

- **Los dos modelos rinden de forma muy parecida.** El mejor valor de la CNN propia es un pico aislado: las épocas vecinas rondan 0,39–0,40 y su pérdida de validación es muy inestable. En las últimas épocas, ResNet-18 es ligeramente superior y más estable (0,42 frente a 0,41), y también queda por delante en test (0,33 frente a 0,31).
- **El sobreajuste aparece pronto.** Ambos modelos alcanzan su mejor F1 en validación en las cinco primeras épocas. Después, el F1 de entrenamiento sigue subiendo hasta 0,63–0,64, mientras que el de validación se estanca en 0,41–0,42 y la pérdida de validación crece.
- **El daño mayor es el principal punto débil.** La CNN propia casi nunca predice esta clase (recall del 0,3 %) y la confunde sobre todo con «sin daño» (63 %) y «daño menor» (35 %). ResNet-18 la reconoce algo más (6,2 %). El daño menor también se confunde a menudo con «sin daño». Los edificios destruidos, cuyo cambio visual es mucho más evidente, se reconocen bien (63–69 %).
- **La validación sobrestima el rendimiento.** El macro-F1 baja de 0,44–0,45 en la mejor época de validación a 0,31–0,33 en el test oculto. Parte de la diferencia se debe a elegir la época con el mismo conjunto de validación; el tamaño de la caída sugiere además que las imágenes de test son más difíciles o tienen otra distribución, algo que no se puede comprobar porque sus etiquetas son ocultas.

**Siguientes pasos:** focal loss o muestreo balanceado para las clases minoritarias, más regularización y augmentation, parches más grandes que incluyan contexto alrededor del edificio, backbones preentrenados en imágenes de teledetección (por ejemplo, [TorchGeo](https://torchgeo.readthedocs.io/)) y early stopping con paciencia en lugar de 25 épocas fijas.

---

## Stack tecnológico

PyTorch, torchvision, scikit-learn, OpenCV, Shapely, tifffile, NumPy, Matplotlib.

---

## Estructura del repositorio

```
xbd-damage-classification/
├── damage_classification.ipynb   # Pipeline completo: datos, augmentation, modelos, entrenamiento y evaluación
├── figures/                      # Matrices de confusión, curvas de entrenamiento y un par de imágenes de ejemplo
├── requirements.txt
├── README.md
└── README.es.md
```

**Cómo ejecutarlo.** Instala las dependencias con `pip install -r requirements.txt` (probado con Python 3.10), coloca el dataset en `data/xBD_UC3M` y ejecuta el notebook de principio a fin. Se recomienda GPU. También puedes abrirlo en Google Colab con el botón del inicio del notebook.

**Datos.** El dataset no se incluye. El notebook usa un subconjunto de xBD proporcionado en la asignatura. El dataset completo está disponible en [xView2](https://www.xview2.org/) bajo su propia licencia (CC BY-NC-SA 4.0).

**Material de la asignatura.** La clase base del dataset (extracción de parches a partir de las anotaciones de xBD), las transformaciones `ToTensor` y `Normalize` y las funciones base de entrenamiento y test fueron proporcionadas por el profesorado. La fusión de 6 canales, las dos arquitecturas, la adaptación de ResNet-18, la data augmentation emparejada, la ponderación de clases, la caché de parches y el análisis de resultados son trabajo del equipo.

---

## Autores

Marcos Morales Tello

**Fernando Martín Arencibia** · [LinkedIn](https://www.linkedin.com/in/fernando-martin-arencibia-477257368/) · [GitHub](https://github.com/fernandomartinarencibia) · [Email](mailto:fernandomartinarencibia@gmail.com)

**Referencia:** Gupta, R. et al. (2019). *xBD: A Dataset for Assessing Building Damage from Satellite Imagery.* arXiv:1911.09296.
