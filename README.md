# Trabajo de Fin de Grado - Repositorio

>**Universidad**: *Universidad de Las Palmas de Gran Canaria*
>
>**Titulación:** *Grado en Ingeniería Informática*
>
>**Título**: *Verificación en escenarios multinivel con presencia de robots asistentes*
>
>**Autor**: Kevin David Rosales Santana
>
>**Tutores**: Modesto Fernando Castrillón Santana y José Javier Lorenzo Navarro.
>
>**AVISO:** El repositorio original tuvo que ser eliminado cuando se publicó el presente repositorio para proteger los derechos de las identidades que se encontraban en versiones antiguas de los cuadernos *Jupyter* y no firmaron el acuerdo de consentimiento de divulgación de sus imágenes (ver apartado de Normativa y Legislación en la Memoria del Trabajo de Fin de Grado).

## Índice

1. [Información del proyecto](#1-Información-del-proyecto)
2. [Navegación del repositorio](#2-Navegación-del-repositorio)
3. [Resultados de los prototipos desarrollados](#3-Resultados-de-los-prototipos-desarrollados)
   - 3.1 [Verificación mediante umbral](#31-Verificación-mediante-umbral)
   - 3.2 [Verificación mediante redes neuronales](#32-Verificación-mediante-redes-neuronales)
   - 3.3 [Uso de los mejores prototipos para los conjuntos de vídeos elegidos](#33-Uso-de-los-mejores-prototipos-para-los-conjuntos-de-vídeos-elegidos)

## 1. Información del proyecto

> La introducción de robots de servicio como asistentes en edificios enfrentándose a problemas diferentes a los existentes actualmente, como el guiado de personas en una única planta, será una realidad en años venideros. Uno de estos es la cooperación bajo interacción hombre-máquina entre varios robots ubicados en diversos pisos de dicho edificio, planteando una comunicación donde uno, localizado en una planta distinta, reconozca a la persona orientada inicialmente en otra entre varias, continuando la colaboración. Para resolver este problema de verificación de identidad se usará el conjunto de datos AveRobot, cuyas condiciones de captura no ideales implican un reto que será encarado estudiando tecnologías biométricas faciales y analizando técnicas de detección, generación y distancia de vectores descriptores o redes neuronales.

Toda la información del proyecto se puede encontrar en la **Memoria del Trabajo de Fin de Grado** entregada junto al enlace de este repositorio y al Manual de Usuario.

Si se tiene cualquier duda, se puede contactar con el autor vía correo electrónico universitario: 

<kevin.rosales101@alu.ulpgc.es>

## 2. Navegación del repositorio

La asistencia necesaria para la correcta navegación y análisis del repositorio se puede encontrar en el [**Manual de Usuario**](./manual-de-usuario.pdf) entregado junto al enlace al presente repositorio.

## 3. Resultados de los prototipos desarrollados

### 3.1 Verificación mediante umbral

- EER: *Equal Error Rate*.
- ACC: *Test accuracy* con el mismo número de pares de vídeos con la misma persona y con distinta persona.

**Localización:** zona del ascensor del Subconjunto de Vídeos B.

| Modelo | Descripción Breve                                            | Mejor umbral |  EER  | Mejor confianza |  ACC   |
| :----: | :----------------------------------------------------------- | :----------: | :---: | :-------------: | :----: |
|   1    | Detección con MTCNN sin normalización + 4 fotogramas máximos por vídeo + *FaceNet* + L2 + distancia euclídea |     1.16     | 0.105 |      31.3%      | 93.09% |
|   2    | Detección con MTCNN con normalización mediante redimensión + 4 fotogramas máximos por vídeo + *FaceNet* + L2 + distancia euclídea |     1.13     | 0.083 |      44.4%      | 94.31% |
|   3    | Detección con MTCNN con normalización básica basada en doble detección + 4 fotogramas máximos por vídeo + *FaceNet* + L2 + distancia euclídea |     1.19     | 0.167 |      44.4%      | 87.81% |
|   4    | Detección con MTCNN con normalización básica basada en recorte estático + 4 fotogramas máximos por vídeo + *FaceNet* + L2 + distancia euclídea |     1.12     | 0.109 |      25.3%      | 93.09% |
|   5    | Detección con MTCNN con normalización mediante redimensión + 4 fotogramas máximos por vídeo + *FaceNet* + L2 + distancia coseno |     0.64     | 0.085 |      44.4%      | 94.31% |
|   6    | Detección con MTCNN con normalización mediante redimensión + 1 fotograma de selección manual por vídeo + *FaceNet* + L2 + distancia euclídea |     1.10     | 0.057 |        -        | 95.94% |
|   7    | Detección con MTCNN con normalización mediante redimensión + 1 fotograma de selección manual por vídeo + *FaceNet* + L2 + distancia coseno |     0.61     | 0.060 |        -        | 95.53% |
|   8    | Detección con DLIB - MMOD con normalización mediante redimensión + 4 fotogramas máximos por vídeo + *FaceNet* + L2 + distancia euclídea |     1.17     | 0.145 |      50.5%      | 90.65% |
|   9    | Detección con DLIB - HOG con normalización mediante redimensión + 4 fotogramas máximos por vídeo + *FaceNet* + L2 + distancia euclídea |     1.13     | 0.107 |      62.6%      | 95.12% |
|   10   | Detección con MTCNN con normalización mediante redimensión + 4 fotogramas máximos por vídeo + *VGGFace2 (ResNet50)* + L2 + distancia euclídea |     1.03     | 0.017 |      13.1%      |  100%  |

Se puede consultar información más detallada en la **sección 7.1.3** de la **Memoria del Trabajo de Fin de Grado.**

### 3.2 Verificación mediante redes neuronales

Todos los errores y tasas de acierto tratan de ser calculados con el mismo número de pares de vídeos con la misma persona y con distinta persona. Todas los prototipos construidos en este tipo de verificación hacen uso de descriptores de *FaceNet*.

- T_LOSS: *Training loss*.
- T_ACC: *Training accuracy*.
- V_LOSS: *Validation loss*.
- V_ACC: *Validation accuracy*.
- LOSS: *Test loss*.
- ACC: *Test accuracy*.

**Localización:** zona del ascensor del Subconjunto de Vídeos B.

| Modelo | Características de la red                                    | T_LOSS | T_ACC  | V_LOSS | V_ACC  | LOSS  |  ACC  |
| :----: | :----------------------------------------------------------- | :----: | :----: | :----: | :----: | :---: | :---: |
|   1    | 4 *embeddings* (2 por vídeo) aportados por MTCNN y DLIB - MMOD | 0.045  | 99.13% | 0.029  | 99.28% | 0.422 | 82.9% |
|   2    | 4 *embeddings* (2 por vídeo) aportados por MTCNN y DLIB - MMOD restados entre sí | 0.013  | 99.77% | 0.007  | 99.87% | 0.358 | 88.5% |
|   3    | 4 *embeddings* (2 por vídeo) aportados por MTCNN y DLIB (MMOD) restados entre sí con valor absoluto | 0.119  | 97.62% | 0.077  | 98.29% | 0.209 | 92.3% |
|   4    | 4 *embeddings* (2 por vídeo) aportados por MTCNN y DLIB (MMOD) restados entre sí con valor absoluto usando un solo fotograma por vídeo. | 0.095  | 99.26% | 0.241  | 89.88% | 0.264 | 90.0% |
|   5    | 2 *embeddings* (1 por vídeo) aportados por MTCNN restados entre sí con valor absoluto | 0.029  | 99.32% | 0.031  | 99.03% | 0.206 | 92.8% |
|   6    | 2 *embeddings* (1 por vídeo) aportados por MTCNN restados entre sí con valor absoluto usando el *Teorema de Kolmogorov* | 0.122  | 95.12% | 0.096  | 96.73% | 0.183 | 93.3% |
|   7    | 2 *embeddings* (1 por vídeo) aportados por MTCNN restados entre sí con valor absoluto usando el *Teorema de Kolmogorov* e introduciendo en una de sus capas la distancia euclídea | 0.039  | 98.80% | 0.215  | 92.34% | 0.417 | 88.2% |
|   8    | 2 *embeddings* (1 por vídeo) aportados por MTCNN restados entre sí con valor absoluto usando el *Teorema de Kolmogorov* alterado e introduciendo en una de sus capas la distancia euclídea | 0.130  | 95.96% | 0.195  | 93.38% | 0.342 | 86.8% |
|   9    | 2 *embeddings* (1 por vídeo) aportados por MTCNN restados entre sí con valor absoluto usando el *Teorema de Kolmogorov* y manteniendo los 2 *embeddings* originales | 0.018  | 99.67% | 0.096  | 96.24% | 0.240 | 92.6% |

Se puede consultar información más detallada en la **sección 7.2.3** de la **Memoria del Trabajo de Fin de Grado.**

### 3.3 Uso de los mejores prototipos para los conjuntos de vídeos elegidos

- LOSS_NN: *Test loss* en el prototipo 6 de verificación mediante redes neuronales.
- ACC_NN: *Test accuracy* en el prototipo 6 de verificación mediante redes neuronales.
- EER: *Equal Error Rate* en el prototipo 10 de verificación mediante umbral de distancia.
- ACC_UMB: *Test Accuracy* en el prototipo 10 de verificación mediante umbral de distancia.

| Subconjunto de vídeos | Cámaras | Localización | Dificultad  | LOSS_NN | ACC_NN |  EER  | ACC_UMB |                Fichero                 |
| :-------------------: | :-----: | :----------: | :---------: | :-----: | :----: | :---: | :-----: | :------------------------------------: |
|           A           | [2,3,8] |  Aleatoria   |   Difícil   |  0.884  | 64.9%  | 0.084 | 95.12%  |    [`random.ipynb`](./random.ipynb)    |
|           B           | [2,5,8] |   Ascensor   |   Normal    |  0.183  | 93.3%  | 0.017 |  100%   | [`main-lift.ipynb`](./main-lift.ipynb) |
|           C           | [2,5,8] |   Pasillo    | Muy difícil |  0.790  | 59.2%  | 0.092 | 91.46%  |  [`corridor.ipynb`](./corridor.ipynb)  |
|           D           | [2,5,8] |   Escalera   | Muy difícil |  0.624  | 65.5%  | 0.103 | 92.68%  |    [`stairs.ipynb`](./stairs.ipynb)    |

Se puede consultar información más detallada en la **sección 7.3** de la **Memoria del Trabajo de Fin de Grado.**