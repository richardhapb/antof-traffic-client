
Existe una API para obtener los feriados, tanto los ya pasados o los futuros. La API se encuentra en https://docs.boostr.cl/reference/holidays

API de feriados periodo: https://api.boostr.cl/holidays/{year}.json (donde `{year}` es el año deseado)
API de feriados futuros: https://api.boostr.cl/holidays.json

---

Incluí en la función "hourly_group" el cálculo de las horas que dura el reporte, con el objetivo de reflejar el tiempo que se encuentra la alerta activa. Para ello, se calcula la diferencia entre la hora de inicio y la hora de término del reporte y se crean reportes ficticios para considerar en el agrupamiento de horas.

---

TODO

Desarrollar modelo random forest distribuido en dos ejes principales:

- [ ] Evaluar probabilidad de que se produzca una alerta
- [ ] Categorizar alerta entre jams y accidents
  - [ ] Se deben equilibrar las categorías, hay menos accidents que jams
- [ ] Predecir calle en donde se producirá el evento
- [ ] Revisar políticas de privacidad de Waze para ver porque no hay datos de usuarios que no han reportado nada.
- [ ] En hourly_group agregar el street para luego no filtrar los datos en modelo ML

