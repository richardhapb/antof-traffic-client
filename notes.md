
Existe una API para obtener los feriados, tanto los ya pasados o los futuros. La API se encuentra en https://docs.boostr.cl/reference/holidays

API de feriados pasados: https://api.boostr.cl/holidays/{year}.json (donde `{year}` es el año deseado)
API de feriados futuros: https://api.boostr.cl/holidays.json

---

Incluí en la función "hourly_group" el cálculo de las horas que dura el reporte, con el objetivo de reflejar el tiempo que se encuentra la alerta activa. Para ello, se calcula la diferencia entre la hora de inicio y la hora de término del reporte y se crean reportes ficticios para considerar en el agrupamiento de horas.
