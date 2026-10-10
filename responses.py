import pandas as pd




# I made a start already, otherwise I could not go on with the state_transition. (Lucas)

responses = {
    "askpreferences": "What kind of restaurant are you looking for?",
    "askfoodtype": "What kind of food would you like?",
    "askarea": "In which area would you like to eat?",
    "askpricerange": "What is your prefered pricerange?",
    "nomatch": "Sorry, there is no restaurant that matches your current preferences.",
    "goodbye": "Goodbye!",
    "null" : "Sorry, I did not understand that. Could you repeat it?",
    "unknownpreference": "Sorry, I am not familiar with this preference. Would you like something else?"
}

reasonings = { 
    "romantic": "The restaurant is romantic, because you can stay for a long time.",
    "touristic": "The restaurant is touristic, because a cheap restaurant with good food attracts tourists.",
    "children": "The restaurant is suitable for children, because you do not have to stay for a long time.",
    "assigned seats": "The waiter decides where you sit because, it is a busy restaurant."
}

contradictions = {
    "romantic": "the restaurant is busy, which is not considered a romantic feature.",
    "touristic": "the restaurant serves Romanian food, which is not considered a touristic feature."
}

def response_generation(system_utterance, restaurant=None, requested=None, food=None, area=None, pricerange=None, suggested_value=None, consequent=None, reasoning=True, contradiction=False):
    
    if system_utterance == "recommendrestaurant" or system_utterance == "recommendrestaurant2" and reasoning == False:
        return f"Based on your preferences I would recommend {restaurant['restaurantname']}."

    if system_utterance == "recommendrestaurant2" and reasoning == True and contradiction==False:
        return f"Based on your preferences I would recommend {restaurant['restaurantname']}.{reasonings[consequent]}"

    if system_utterance == "recommendrestaurant2" and reasoning == True and contradiction==True:
        return f"Based on your preferences I would recommend {restaurant['restaurantname']}.{reasonings[consequent]} However, {contradictions[consequent]}"

    if system_utterance == "askadditionalrequirements":
        return f"I found some restaurants matching your preferences. Do you have any additional requirements?"
        
    if system_utterance == "provide":

        restaurant_name = restaurant["restaurantname"]
        value = restaurant[requested]

        if pd.isna(value):
            return f"The {requested} of {restaurant_name} is unknown."

        if requested == "phone":
            return f"The phone number of {restaurant_name} is {value}."

        elif requested == "addr":
            return f"{restaurant_name} is located at {value}."

        elif requested == "postcode":
            return f"The postcode of {restaurant_name} is {value}."
        
    
    if system_utterance == "askconfirmation":
        return f"Did you mean {suggested_value}?"


    if system_utterance == "nomatch2":
        if consequent == "romantic" or consequent == "touristic":
            return f"None of the restaurants matching your preferences are {consequent}."
        else:
            return f"None of the restaurants matching your preferences are suitable for {consequent}."
    
    
    if system_utterance == "confirmproperty":
        actual_value = restaurant[requested]

        if suggested_value == actual_value:
            return "Yes, that's correct."
        else:
            return f"No, the {requested} is actually {actual_value}."

    

    if system_utterance in responses:
        return responses[system_utterance]

    return "ERROR in response_generation"
