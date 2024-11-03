from waze.events import Events

alerts = Events(filename="data/alerts.json", table_name="alerts")
jams = Events(filename="data/jams.json", table_name="jams")

alerts.read_file()
jams.read_file()

alerts.insert_to_db(review_mode="all")
jams.insert_to_db(review_mode="all")
