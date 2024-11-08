from utils import utils
from analytics.grouper import Grouper


alerts = utils.load_data("alerts").to_gdf()[["type", "pubMillis", "geometry"]]

g = Grouper()
g.group(alerts, (10, 20))

total_data = utils.generate_no_events(
    g.data.drop("geometry", axis=1).rename(columns={"pubMillis": "inicio"})
)

print(total_data.shape)
