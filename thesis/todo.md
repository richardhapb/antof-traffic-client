- [ ] Gernerate graph from RPI data
- [ ] Verify period of time for current available graphics
- [ ] Evaluate factibility of updating graphics
- [ ] Include statistics data

Comentarios preliminares:
Si el documento se esta redactando el LaTeX, compartir acceso para revisar directamente desde el código!
Llevar Formato de tesis según estructura indicada en ramo Proyecto de Titulo.
Este tipo de documentos debe ser redactado en tercera persona gramatical. Revisar y corregir esto a lo largo del manuscrito.

RESUMEN
Citar Plataforma Waze cities
El resumen debe proporcionar datos duros del trabajo desarrollado
No se recomienda incluir citas en el resumen

INTRODUCCION
Cada sección debe comenzar en una nueva hoja. Sección demasiado escueta... recomiendo la siguiente organización... Puede omitir las subsecciones que estime convenientes (o fusionarlas)!
1. Contextualización del problema: Breve descripción del crecimiento urbano de Antofagasta; Problemáticas actuales del tráfico vehicular en la ciudad; Rol de las tecnologías de información y comunicación (TIC) en la movilidad urbana.
2. Justificación del estudio: Importancia de contar con datos en tiempo real para la gestión del tráfico; Limitaciones de los sistemas tradicionales de monitoreo vehicular; Potencial de plataformas colaborativas como Waze en ciudades intermedias.
3. Formulación del problema: ¿Cómo puede utilizarse la información de Waze Cities para modelar y predecir el comportamiento del tráfico vehicular en Antofagasta?
4. Objetivos: General y Específicos
5. Alcance y delimitaciones: Periodo y área geográfica de estudio; Límites tecnológicos y de acceso a datos; Exclusión de variables como transporte público, condiciones climáticas, etc
6. Metodología general: Descripción breve de herramientas como GeoPandas, librerías de series temporales, y modelos de machine learning.
7. Estructura del documento: Breve descripción de los capítulos que componen la tesis.

MARCO TEORICO
Desagregar capitulo... Estructura actual demasiado acotada!
Jamás se explican los métodos utilizados.
Propongo la siguiente estructura... Puede omitir las subsecciones que estime convenientes (o fusionarlas)!
1. Sistemas de transporte urbano y congestión: Características del sistema vial urbano; Causas y consecuencias de la congestión vehicular; Indicadores comunes para evaluar el rendimiento del tráfico.
2. Ciencia de datos en el análisis de movilidad: Tipos de datos utilizados en movilidad urbana (GPS, sensores, crowdsourcing); Ventajas del uso de datos colaborativos como los de Waze; Estudios previos que han utilizado Waze Cities Data.
3. Análisis geoespacial aplicado al tráfico: Fundamentos del análisis geográfico aplicado a la movilidad; Uso de herramientas como GeoPandas, shapefiles, mapas de calor y sistemas de información geográfica (SIG); Ejemplos de visualización de eventos de tráfico.
4. Modelado de series temporales: Conceptos básicos (tendencia, estacionalidad, ruido); Técnicas comunes (ARIMA, Prophet, modelos de suavizado exponencial); Aplicaciones en tráfico vehicular.
5. Aprendizaje automático en predicción de tráfico: Breve introducción al machine learning supervisado; Algoritmos relevantes (Random Forest, Gradient Boosting, Redes Neuronales); Ventajas frente a modelos tradicionales; Aplicaciones en estudios urbanos.
6. Aportes de investigaciones previas: Revisión de trabajos clave (como Van Lint et al., 2005 y Barceló & Casas, 2005). Comparación de metodologías y resultados; Identificación de vacíos que la presente investigación aborda.

DESARROLLO
Sección 3.3: R2 mal usado. El coeficiente de determinación no es lo mismo que la correlación. Sugiero calcular la correlación Pearson o Stearman, y argumentar en base todos los estadísticos considerados.
Recuerda que todos los indicadores que utilizaras aca debes haberlos introducido/explicado en la sección de Marco Teórico, quizás alguna Sección o Subsección llamada Indicadores de Bondad de Ajuste.
Figuras 7 y 8: Todos los objetos deben insertarse después de que son nombrados en el manuscrito. Revisar esto a lo largo del manuscrito!
Sección 3.7: Se nombran los algoritmos de clasificación utilizados, pero dichos modelos no fueron explicados en el marco teórico.
Sección 3.9: Evitar salirse de los márgenes.
Sección 3.11: Revisar cita de Dash... (Peña, 2024); las herramientas como las librerías APScheduler tienen que ser citadas. Las demás librerías también deben citarse. Revisar esto a lo largo del manuscrito.
Desarrollo demasiado acotado. Complementar!
Sugiero algo similar a lo indicado a continuación... Puede omitir las subsecciones que estime convenientes (o fusionarlas)!
1. Descripción de la fuente de datos: Plataforma Waze Cities y su funcionamiento; Tipos de datos obtenidos (alertas, tráfico, accidentes, congestión, etc.); Alcance espacial y temporal de los datos utilizados; Consideraciones éticas y de privacidad.
2. Preprocesamiento y limpieza de datos: Carga y exploración inicial de los datos; Identificación y tratamiento de datos faltantes o erróneos; Normalización y estructuración de variables; Conversión de coordenadas y formatos geoespaciales.
3. Análisis exploratorio de datos (EDA): Distribución temporal de los eventos (por hora, día, semana); Identificación de zonas críticas mediante visualización geográfica; Detección de patrones de congestión según tipo de evento; Generación de mapas de calor y mapas interactivos
4. Análisis geoespacial: Integración con capas urbanas (vialidad, comunas, sectores); Uso de la librería GeoPandas y herramientas complementarias (Shapely, Folium, etc.); Delimitación de zonas de alto riesgo o congestión; Correlaciones espaciales (por ejemplo, densidad de eventos vs. tipo de vía)
5. Modelamiento de series temporales: Preparación de la serie temporal (resampleo, agregación); Evaluación de estacionalidad y tendencias; Aplicación de modelos clásicos (ARIMA, ETS, Prophet); Métricas de evaluación del modelo (RMSE, MAE, etc.). Visualización de predicciones.
6. Aplicación de modelos de aprendizaje automático: Selección de variables predictoras; División de datos en entrenamiento y prueba; Modelos implementados (Random Forest, Gradient Boosting, etc.); Comparación de modelos y ajuste de hiperparámetros; Interpretación de resultados (importancia de variables, explicabilidad).
7. Resultados integrados y visualización final: Cuadro resumen de hallazgos clave; Visualización interactiva de rutas críticas y predicciones; Recomendaciones para la planificación urbana y gestión del tráfico; Discusión sobre la utilidad práctica de la herramienta desarrollada

RESULTADOS -- recomendaría cambiar el nombre del capitulo por DISCUSIONES
Resultados demasiado acotados... Considerar la siguiente desagregación como ejemplo! No considerar las que no estimes pertinentes...
1. Análisis de patrones espaciales y temporales del tráfico: Principales zonas de congestión e incidentes detectadas; Variaciones horarias y diarias en la ocurrencia de eventos; Comparación con el conocimiento empírico/local del tránsito en Antofagasta; Posibles factores estructurales o contextuales que explican los patrones observados
2. Discusión sobre el modelamiento predictivo: Desempeño de los modelos de series temporales; Precisión y capacidad predictiva de los modelos de aprendizaje automático; Comparación entre métodos clásicos y de machine learning; Interpretación de variables predictoras relevantes y su importancia; Ventajas y limitaciones de los modelos aplicados en contextos urbanos reales
3. Aporte del análisis geoespacial en la toma de decisiones: Visualizaciones clave para la comprensión del tráfico urbano; Valor añadido del enfoque espacial frente a análisis meramente tabulares; Potencial uso por parte de autoridades locales o instituciones de transporte; Limitaciones del análisis geoespacial en entornos urbanos incompletos o no mapeados
4. Implicancias prácticas y futuras aplicaciones: Recomendaciones para la planificación urbana y vial en Antofagasta; Posibilidad de escalabilidad del modelo a otras ciudades chilenas; Integración con sistemas de gestión del tráfico y políticas públicas; Aplicación en situaciones de emergencia, planificación de eventos masivos o contingencias climáticas
5. Limitaciones del estudio Restricciones técnicas, computacionales o de disponibilidad de datos; Problemas encontrados en la implementación de modelos; Consideraciones sobre la generalización de los resultados

REFERENCIAS
Complementar la bibliografía. Cantidad de fuentes citadas demasiado acotadas!
