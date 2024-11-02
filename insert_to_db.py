from waze.events import Events

alerts = Events(filename="data/alerts.json", table_name="alerts")
jams = Events(filename="data/jams.json", table_name="jams")

alerts.insert_to_db(only_review_not_ended=False)
jams.insert_to_db(only_review_not_ended=False)
