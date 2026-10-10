import random
import pandas as pd

restaurants = pd.read_csv("Data/restaurant_info.csv")

def restaurant_lookup(area, food, pricerange):

    matches = restaurants

    if area != "dontcare":
        matches = matches[matches["area"] == area]

    if food != "dontcare":
        matches = matches[matches["food"] == food]

    if pricerange != "dontcare":
        matches = matches[matches["pricerange"] == pricerange]
        
        
    matches = matches.to_dict("records")

    if len(matches) == 0:
        return None, []
    
    recommendation = random.choice(matches)
    
    alternatives = [restaurant for restaurant in matches if restaurant != recommendation]
        
    return recommendation, alternatives

RESTAURANT_FILE2 = "Data/restaurant_info_extended.csv"

restaurants_extended = pd.read_csv(RESTAURANT_FILE2)

def antecedent_rules(consequent):
    if consequent == "touristic":
        true={"pricerange": "cheap",
             "food quality": "good"}
        false= {"food": "romanian"}
    elif consequent == "assigned seats":
        true={"crowdedness": "busy"}
        false= {}
    elif consequent == "children":
        true={}
        false={"length of stay": "long"}
    elif consequent == "romantic":
        true={"length of stay": "long"} 
        false={"crowdedness": "busy"}
    return true, false

def reasoning_component(restaurants_options, consequent, requested_value=True):
    true, false = antecedent_rules(consequent)

    matches = []
    contradictions = []
    recommendation = None
    contra = False
    alternatives = []

    for restaurant_option in restaurants_options:
        restaurant = restaurants_extended[restaurants_extended["restaurantname"] == restaurant_option["restaurantname"]].iloc[0]

        check_true = (bool(true) and 
                          all(restaurant[attribute] == value
                              for attribute, value in true.items()))

        check_false = (bool(false) and 
                          all(restaurant[attribute] == value
                              for attribute, value in false.items()))

        if not true and false and requested_value == True:
            # No positive rule exists: match restaurants
            # that do not satisfy the negative rule.
            if not check_false:
                matches.append(restaurant_option)

        if check_true and not(check_false) and requested_value == True:
            matches.append(restaurant_option)

        elif check_false and not(check_true) and requested_value == False:
            matches.append(restaurant_option)
        
        elif check_true and check_false:
            contradictions.append(restaurant_option)

    if matches:
        recommendation=random.choice(matches)
        alternatives = [restaurant for restaurant in matches if restaurant != recommendation]

    elif contradictions:
        contra = True
        recommendation=random.choice(contradictions)
        alternatives = [restaurant for restaurant in contradictions if restaurant != recommendation]
        
    return recommendation, alternatives, contra
