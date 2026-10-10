import pandas as pd

RESTAURANT_FILE = "Data/restaurant_info.csv"

restaurants = pd.read_csv(RESTAURANT_FILE)

#print(restaurants.columns)
#restaurants.head()

areas = restaurants["area"].dropna().unique().tolist()
foods = restaurants["food"].dropna().unique().tolist()
price_ranges = restaurants["pricerange"].dropna().unique().tolist()
consequents=["touristic", "assigned seats", "children", "romantic"]

#print(areas)
#print(price_ranges)
#print(foods)
