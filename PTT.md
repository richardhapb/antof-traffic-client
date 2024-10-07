
## Abstract

Debido al 
A través del uso de datos otorgados por Waze, estos datos están basados en eventos, ya que Waze no tiene datos de los vehículos que no han reportado algun evento, al menos no los tiene de forma pública. Se genera un análisis exploratorio de los datos, y posteriormente se ajustará un modelo de Machine Leraning para intentar predecir el comportamiento de los eventos de tráfico.

Waze no entrega un dataset donde estén almacenados todos los datos de forma histórica, por lo que se genera un script que tenga todos los datos relacionados.

Los datos se conforman de dos estructuras principales:

> INSERTAR DIAGRAMA 

Donde:

- `alerts` contiene la base de datos de las alertas reportadas por los usuarios, la información es entregada vía JSON y tiene la siguiente estructura:
  
<table class="nice-table">
    <tbody>
        <tr>
        <td>
        <p><strong data-outlined="false" class="">Element</strong></p>
        </td>
        <td>
        <p><strong>Value</strong></p>
        </td>
        <td>
        <p><strong data-outlined="false" class="">Description</strong></p>
        </td>
        </tr>
        <tr>
        <td>
        <p data-outlined="false" class="">pubMillis</p>
        </td>
        <td>
        <p data-outlined="false" class="">Timestamp</p>
        </td>
        <td>
        <p data-outlined="false" class="">Publication date (Unix time – milliseconds since epoch)</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>location</p>
        </td>
        <td>
        <p>Coordinates</p>
        </td>
        <td>
        <p>Location per report (X Y - Long-lat)</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>uuid&nbsp;</p>
        </td>
        <td>
        <p>String</p>
        </td>
        <td>
        <p>Unique system ID&nbsp;</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>magvar</p>
        </td>
        <td>
        <p>Integer (0-359)</p>
        </td>
        <td>
        <p>Event direction (Driver heading at report time. 0 degrees at North, according to the driver’s device)</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>type</p>
        </td>
        <td>
        <p>See alert type table</p>
        </td>
        <td>
        <p>Event type</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>subtype</p>
        </td>
        <td>
        <p>See alert sub types table</p>
        </td>
        <td>
        <p>Event sub type - depends on atof parameter</p>
        </td>
        </tr>
        <tr>
        <td>
        <p data-outlined="false" class="">reportDescription</p>
        </td>
        <td>
        <p data-outlined="false" class="">String</p>
        </td>
        <td>
        <p data-outlined="false" class="">Report description (supplied when available)</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>street</p>
        </td>
        <td>
        <p>String</p>
        </td>
        <td>
        <p>Street name (as is written in database, no canonical form, may be null)</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>city</p>
        </td>
        <td>
        <p>String</p>
        </td>
        <td>
        <p>City and state name [City, State] in case both are available, [State] if not associated with a city. (supplied when available)</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>country</p>
        </td>
        <td>
        <p>String</p>
        </td>
        <td>
        <p><span>(see two letters codes in &nbsp;<a href="http://en.wikipedia.org/wiki/ISO_3166-1"><u>http://en.wikipedia.org/wiki/ISO_3166-1</u></a>)&nbsp;</span></p>
        </td>
        </tr>
        <tr>
        <td>
        <p>roadType</p>
        </td>
        <td>
        <p>Integer</p>
        </td>
        <td>
        <p><span>Road type (see <a href="#Roadtypes" rel="noopener"><u>road types</u></a>)</span></p>
        </td>
        </tr>
        <tr>
        <td>
        <p>reportRating</p>
        </td>
        <td>
        <p>Integer</p>
        </td>
        <td>
        <p>User rank between 1-6 ( 6 = high ranked user)</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>jamUuid</p>
        </td>
        <td>
        <p>string</p>
        </td>
        <td>
        <p>If the alert is connected to a jam - jam ID</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>Reliability&nbsp;</p>
        </td>
        <td>
        <p>0-10</p>
        </td>
        <td>
        <p>Reliability score based on user reactions and reporter level</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>confidence</p>
        </td>
        <td>
        <p>0-10</p>
        </td>
        <td>
        <p>Confidence score based on user reactions</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>reportByMunicipalityUser</p>
        </td>
        <td>
        <p>Boolean</p>
        </td>
        <td>
        <p>Alert reported by municipality user (partner) Optional.</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>nThumbsUp</p>
        </td>
        <td>
        <p>integer</p>
        </td>
        <td>
        <p>Number of thumbs up by users</p>
        </td>
        </tr>
    </tbody>
</table>

- `jams` son los datos que Waze genera automáticamente cuando existe una congestión de tráfico, almacena los trazos por los cuales se mueve la congestión. Se estructura de la siguiente forma:

<table class="nice-table">
    <tbody>
        <tr>
        <td>
        <p><strong>Element</strong></p>
        </td>
        <td>
        <p><strong>Value</strong></p>
        </td>
        <td>
        <p><strong>Description</strong></p>
        </td>
        </tr>
        <tr>
        <td>
        <p>pubMillis</p>
        </td>
        <td>
        <p>Timestamp</p>
        </td>
        <td>
        <p>Publication date (Unix time – milliseconds since epoch)</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>type</p>
        </td>
        <td>
        <p>String</p>
        </td>
        <td>
        <p>TRAFFIC_JAM&nbsp;</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>line</p>
        </td>
        <td>
        <p>List of Longitude and Latitude coordinates</p>
        </td>
        <td>
        <p>Traffic jam line string (supplied when available)</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>speed</p>
        </td>
        <td>
        <p>Float</p>
        </td>
        <td>
        <p>Current average speed on jammed segments in meters/seconds</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>speedKPH</p>
        </td>
        <td>
        <p>Float</p>
        </td>
        <td>
        <p>Current average speed on jammed segments in Km/h</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>length</p>
        </td>
        <td>
        <p>Integer</p>
        </td>
        <td>
        <p>Jam length in meters</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>delay</p>
        </td>
        <td>
        <p>Integer</p>
        </td>
        <td>
        <p>Delay of jam compared to free flow speed, in seconds (in case of block, -1)</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>street</p>
        </td>
        <td>
        <p>String</p>
        </td>
        <td>
        <p>Street name (as is written in database, no canonical form. (supplied when available)</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>city</p>
        </td>
        <td>
        <p>String</p>
        </td>
        <td>
        <p>City and state name [City, State] in case both are available, [State] if not associated with a city. (supplied when available)</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>country</p>
        </td>
        <td>
        <p>String</p>
        </td>
        <td>
        <p><span>available on EU (world) server (see two letters codes in &nbsp;<a href="http://en.wikipedia.org/wiki/ISO_3166-1"><u>http://en.wikipedia.org/wiki/ISO_3166-1</u></a>)</span></p>
        </td>
        </tr>
        <tr>
        <td>
        <p>roadType</p>
        </td>
        <td>
        <p>Integer</p>
        </td>
        <td>
        <p><span>Road type (see <a href="#Roadtypes" rel="noopener"><u>road types</u></a>&nbsp;)</span></p>
        </td>
        </tr>
        <tr>
        <td>
        <p>startNode</p>
        </td>
        <td>
        <p>String</p>
        </td>
        <td>
        <p>Nearest Junction/steet/city to jam start (supplied when available)</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>endNode</p>
        </td>
        <td>
        <p>String</p>
        </td>
        <td>
        <p>Nearest Junction/steet/city to jam end (supplied when available)</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>level</p>
        </td>
        <td>
        <p>0 - 5</p>
        </td>
        <td>
        <p>Traffic congestion level (0 = free flow 5 = blocked).</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>uuid</p>
        </td>
        <td>
        <p>Long integer</p>
        </td>
        <td>
        <p>Unique jam ID</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>turnLine</p>
        </td>
        <td>
        <p>Coordinates</p>
        </td>
        <td>
        <p>A set of coordinates of a turn - only when the jam is in a turn (supplied when available)</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>turnType</p>
        </td>
        <td>
        <p>String</p>
        </td>
        <td>
        <p>What kind of turn is it - left, right, exit R or L, continue straight or NONE (no info) (supplied when available)</p>
        </td>
        </tr>
        <tr>
        <td>
        <p>blockingAlertUuid</p>
        </td>
        <td>
        <p>string</p>
        </td>
        <td>
        <p>if the jam is connected to a block (see alerts)&nbsp;</p>
        </td>
        </tr>
    </tbody>
</table>

Fuente: [Waze cities reference](https://support.google.com/waze/partners/answer/13458165?hl=en&ref_topic=10616686&sjid=6478162241018921516-SA#zippy=%2Ctraffic-alerts)

## Metodología

Se seguirán los siguientes pasos:

  1. Generación de script para la captura de datos
  2. Montaje de script en un servidor
  3. Análisis exploratorio de datos
  4. Análisis descriptivo
  5. Análisis inferencial
  6. Selección modelo Machine Learning
  7. Entrenamiento modelo
  8. Implementación modelo Machine Learning
