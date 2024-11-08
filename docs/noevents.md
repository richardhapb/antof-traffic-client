:::mermaid
graph TD

    A[Start] --> B[define noevents array and initialize happen to 1]
    B --> C[For each in range -> inicio, fin, 1060]
    C --> D{if data->inicio == i}
    D --> |yes| E[g = groups in data->inicio and t in data->types]
    D --> F[noevents.append -> groups not in element with happen 0]
    E --> F
    F --> G[noevents -> types not in element with happen 0 for each group]
    G-->C
:::

