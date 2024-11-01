::: mermaid
graph TD
    title[<u>WAZE'S API DATA FLOW</u>]
    style title fill:#FFF,stroke:#000,stroke-width:2px, font-size: 20px, font-weight: bold,color:#000

    A[Start] --> B[Initialize WazeAPI]
    B --> F[Fetch data from WazeAPI]
    F --> G{Check if data is empty}
    G -->|No| H[Create Events object]
    G -->|Yes| LOOP[Wait 5 minutes]
    H --> M[Clean events data]
    M --> Q[Fetch last 24 hours events from database]
    Q --> T[Merge events that not are in the database - uuid as key]
    T --> U[Update end reports timestamp for events that not are in the API data but are in the database]
    U --> V[Insert new events to database]
    V --> LOOP
    LOOP --> F
:::