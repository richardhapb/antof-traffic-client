``` mermaid
flowchart TB
    title[<u>WAZE's API DATA FLOW</u>]
    style title fill:#FFF,stroke:#000,stroke-width:2px, font-size: 20px, font-weight: bold,color:#000

    A[Start] --> B[Initialize WazeAPI]
    B --> F[Fetch data from WazeAPI]
    F --> G{Check if data is empty}
    G -->|No| H[Create Events object]
    G -->|Yes| LOOP[Wait 5 minutes]
    H --> M[Clean events data]
    M --> Q[Fetch last 24 hours events from database]
    Q --> U[Merge events that not are in the database - uuid as key]
    U --> T[Insert new events to database]
    T --> V[Update end reports timestamp for events\nthat not are in the API data but are in the\ndatabase without end timestamp]
    V --> LOOP
    LOOP --> F
```

