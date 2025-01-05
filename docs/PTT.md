





PRESENTACIÓN PROYECTO DE TRABAJO DE TÍTULO


INGENIERÍA CIVIL INDUSTRIAL, MENCIÓN GESTIÓN

“Análisis Basado en Datos del Comportamiento del Tráfico Vehicular en Antofagasta: Un Enfoque a partir de Reportes de Conductores”





Alumno/as           : Richard Peña Bonifaz
Profesor Patrocinante   : Manuel Saldaña Pino
Profesor Colaborador    : Luis Ayala Alcázar
Fecha de Presentación   :


1.- TÍTULO DEL TEMA:

“Análisis Basado en Datos del Comportamiento del Tráfico Vehicular en Antofagasta: Un Enfoque a partir de Reportes de Conductores”


2. OPCION DE TRABAJO DE TÍTULO A DESARROLLAR: Memoria

3.- RESUMEN

Este proyecto tiene como objetivo analizar el comportamiento del tráfico vehicular en la ciudad de Antofagasta utilizando datos de la plataforma Waze Cities. Waze, a través de su comunidad de usuarios, ofrece información en tiempo real que permite obtener una visión detallada de los eventos de tráfico que ocurren en la ciudad. Esta investigación busca generar información relevante que contribuya a la gestión del tráfico, apoyando la toma de decisiones por parte de las autoridades locales para mejorar la seguridad vial y la eficiencia del flujo vehicular. A través del análisis de estos datos, se podrán identificar patrones y tendencias que, cuando se integren en la planificación urbana, ayudarán a optimizar las rutas críticas de la ciudad, reducir la congestión y disminuir la probabilidad de accidentes.

En este estudio se aplicarán técnicas de análisis de datos y geoespaciales con GeoPandas y series temporales, proporcionando una visualización clara y comprensible para las autoridades de gestión vial. Asimismo, se explorará la viabilidad de aplicar técnicas de aprendizaje automático para realizar predicciones de tráfico, con el fin de anticiparse a problemas y contribuir a un desarrollo urbano más eficiente y seguro (Barceló & Casas, 2005; Van Lint et al., 2005).

4.- DESCRIPCIÓN DEL PROBLEMA:

Antofagasta, una ciudad con más de 106,000 vehículos en circulación (Comisión Nacional de Seguridad de Tránsito, 2023), enfrenta serios desafíos en la gestión de su tráfico vehicular. En el año 2023, se registraron 1,715 accidentes, resultando en 31 fallecidos y 102 heridos graves (Comisión Nacional de Seguridad de Tránsito, 2023). La infraestructura vial limitada de la ciudad y la concentración de vehículos en pocas arterias principales agravan la situación, generando congestiones y aumentando el riesgo de accidentes. A pesar de la magnitud de estos problemas, actualmente no existen sistemas de monitoreo en tiempo real que permitan gestionar de manera proactiva el tráfico en la ciudad. Por tanto, se requiere aprovechar fuentes de datos alternativas, como Waze, para recolectar información y mejorar la toma de decisiones relacionadas con el tráfico (Chen et al., 2015).



5.- OBJETIVOS:
5.1.- GENERAL:

Desarrollar un análisis exhaustivo del comportamiento del tráfico en la ciudad de Antofagasta basado en los eventos reportados por conductores en la plataforma Waze. El objetivo final es proporcionar información relevante para la gestión del tráfico, optimizando la toma de decisiones en términos de seguridad vial y eficiencia en el flujo vehicular.


5.2.- ESPECÍFICOS:

    •   Obtener datos suficientes y representativos sobre los eventos de tráfico en la ciudad de Antofagasta mediante la API de Waze Cities.
    •   Realizar un análisis descriptivo de los datos obtenidos para identificar patrones y tendencias relevantes que afecten el comportamiento del tráfico.
    •   Determinar los factores clave que inciden en la seguridad vial y la eficiencia del tráfico en la ciudad.
    •   Proporcionar información útil y visualmente comprensible para las autoridades encargadas de la gestión vial, que facilite la implementación de políticas y acciones basadas en datos (Auld & Mohammadian, 2009).

6. METODOLOGÍA.
6.1 MARCO TEORICO

El tráfico vehicular en áreas urbanas presenta un comportamiento complejo e impredecible, lo que dificulta su gestión efectiva. Sin embargo, el avance de las tecnologías móviles y la proliferación de aplicaciones como Waze permiten contar con datos en tiempo real generados por los mismos usuarios. Este proyecto se apoyará en técnicas de análisis de datos para transformar esta información en herramientas útiles para la gestión vial. El uso de datos geoespaciales y la capacidad de automatizar los procesos de recolección, análisis y visualización de datos ofrecen una solución costo-efectiva para mejorar la planificación del tráfico (Barceló & Casas, 2005).

6.2 METODOLOGÍA

Una de las principales dificultades a la hora de analizar el comportamiento de determinados fenómenos es la ausencia de datos. Por lo que se planteará una estrategia de recolección de datos que permita obtener conclusiones con un determinado grado de certeza. Se recopilarán datos de la API de Waze Cities, que proporciona información de los eventos activos al momento de la consulta, por lo que para poder obtener una cantidad relevante de datos se implementará un servidor que recopile y almacene información para el análisis.

Se desarrollará un análisis de los datos con la capacidad de poder recalcularse a medida que se continuen agregando nuevos elementos a la base de datos. Se llevará a cabo un análisis geoespacial que permita identificar puntos de interés, como las vías principales, las calles principales y las zonas de tráfico. Este análisis se realizará utilizando el paquete geopandas de Python, que permite realizar operaciones de análisis geoespaciales. Se analizarán series temporales de datos para identificar patrones y tendencias en el comportamiento del tráfico, y se utilizarán técnicas de visualización para presentar los resultados para identificar estacionalidades de comportamiento.

Se implementará un pipeline de datos para poder realizar el análisis, que incluirá la recolección de datos, la limpieza y transformación, la visualización y la análisis descriptivo. Se evaluará la posibilidad de utilizar técnicas de aprendizaje automático (machine learning) para predecir el comportamiento del tráfico en el futuro y poder tomar decisiones en consecuencia.

7. BIBLIOGRAFÍAS Y FUENTES DE LA INFORMACIÓN.
7.1 FUENTES DE INFORMACIÓN

    •   Auld, J., & Mohammadian, A. (2009). Framework for modeling activity planning as a sequence of planned activities. Transportation Research Part B: Methodological, 43(6), 924-937.
    •   Barceló, J., & Casas, J. (2005). Dynamic network simulation with AIMSUN. Simulation approaches in transportation analysis, 57-98.
    •   Van Lint, J. W. C., Hoogendoorn, S. P., & Van Zuylen, H. J. (2005). Accurate freeway travel time prediction with state-space neural networks under missing data. Transportation Research Part C: Emerging Technologies, 13(5-6), 347-369.
    •   Chen, C., Zhang, J., & Antoniou, C. (2015). Data-driven approaches to traffic prediction and management. Transportation Research Procedia, 10, 12-24.
    •   Goodall N. Lee E. (Tina) 2019. Comparison of Waze crash and disabled vehicle records with video evidence. Transportation Research Interdisciplinary Perspectives. https://doi.org/10.1016/j.trip.2019.100019.
    •   Berhanu Y. Schröder D. Teklu Wodajoc B. Alemayehua E. (2024). Machine learning for predictions of road traffic accidents and spatial network analysis for safe routing on accident and congestion-prone road networks. Results in Engineering. https://doi.org/10.1016/j.rineng.2024.102737.

8. CRONOGRAMA TENTATIVO


9.  IDENTIFICACIÓN
9.1 NOMBRE DEL ALUMNO
    Nombre  :
    RUT :
    Carrera :
    Ciudad  :
    Año ingreso :
    Año egreso  :
    Correo electrónico  :
    Teléfono contacto   :

9.2 NOMBRE PROFESOR PATROCINANTE Y COLABORADOR
    Profesor Patrocinante   :   Luis Ayala Alcázar
    RUT :   12.613.447-9
    Profesión   :   Ingeniero Civil Industrial
    Cargo   :   Jefe de Carrera
    Institución :   Universidad Arturo Prat

    Profesor Colaborador    :   Manuel Saldaña pino
    RUT :   18.722.588-4
    Profesión   :   Ingeniero Civil Industrial
    Cargo   :   Académico, investigador
    Institución :   Universidad Arturo Prat


