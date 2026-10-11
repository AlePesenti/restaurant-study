from slot_extraction import (keyword_slot_extractor, levenshtein_slot_extractor,
                             semantic_slot_extractor, request_extractor)
from responses import response_generation
from reasoning import reasoning_component, restaurant_lookup


def create_dialog_state():
    dialog_state =  {
        "state": 1,
        "area": None,
        "food": None,
        "pricerange": None,
        "restaurant": None,
        "alternatives": [],
        "last_response": None,
        "confirmation" : None,
        "consequent" : None
    }
    
    return dialog_state


create_dialog_state()

# state transition function

def state_transition(dialog_state, utterance, dialog_act, fallback="levenshtein"):
    
    current_state = dialog_state["state"]
    next_dialog_state = current_state
    system_utterance = None
    
    
    # Confirmation part
    if current_state == 6 and dialog_state["confirmation"] is not None:
        slot = dialog_state["confirmation"]["slot"]
        value = dialog_state["confirmation"]["value"]

        if dialog_act == "affirm":
            dialog_state[slot] = value
            dialog_state["confirmation"] = None

        elif dialog_act in ["negate", "deny"]:
            dialog_state["confirmation"] = None
                
        else:
            system_utterance = dialog_state["last_response"]
    
    
    
    if dialog_act == "restart":
        dialog_state = create_dialog_state()
        next_dialog_state = 2
        system_utterance = response_generation("askpreferences")

    elif dialog_act in ["bye", "thankyou"]:        
        next_dialog_state = 11  
        system_utterance = response_generation("goodbye")

    elif dialog_act == "repeat":
        system_utterance = dialog_state["last_response"]
    
    elif current_state == 1 and dialog_act == "hello":
        next_dialog_state = 2
        system_utterance = response_generation("askpreferences")

        
    elif ((current_state in [1, 2, 3, 4, 5, 7] and dialog_act in ["inform", "reqalts"]) 
           or (current_state in [8, 9] and dialog_act == "inform")):  # Fix for reference dialog 2: U: another one in the west

        # Keyword slot extractor
        slots = keyword_slot_extractor(utterance, current_state)

        if slots["food"] is not None:
            dialog_state["food"] = slots["food"]

        if slots["area"] is not None:
            dialog_state["area"] = slots["area"]

        if slots["pricerange"] is not None:
            dialog_state["pricerange"] = slots["pricerange"]
            
            
        # Fallback levenshtein extractor
        if fallback == "levenshtein":
            fallback_slots = levenshtein_slot_extractor(utterance)

        # Fallback semantic extractor
        elif fallback == "semantic":
            fallback_slots = semantic_slot_extractor(utterance)
            
            
        # Confirmation part
        for slot in ["food", "area", "pricerange"]:
            if (slots[slot] is None and dialog_state[slot] is None and fallback_slots[slot] is not None):
                
                dialog_state["confirmation"] = {"slot": slot, "value": fallback_slots[slot]}

                next_dialog_state = 6
                system_utterance = response_generation("askconfirmation", suggested_value=fallback_slots[slot])

                dialog_state["state"] = next_dialog_state
                dialog_state["last_response"] = system_utterance

                return dialog_state, system_utterance
            
        
        # If an "inform" occurs but the preference does not gets gound. Reference_dialog 6: "I would like some Swedish food."
        if (all(value is None for value in slots.values()) and all(value is None for value in fallback_slots.values())):
            system_utterance = response_generation("unknownpreference")
            
            
    elif current_state in [8, 9, 10] and dialog_act in ["reqalts", "reqmore"]:
        
        if len(dialog_state["alternatives"]) > 0:
            recommendation = dialog_state["alternatives"].pop(0)

            dialog_state["restaurant"] = recommendation

            next_dialog_state = 8
            system_utterance = response_generation("recommendrestaurant", restaurant=recommendation)

        else:
            next_dialog_state = 7
            system_utterance = response_generation("nomatch")      

            
    elif current_state in [8, 9, 10] and dialog_act == "request":

        requested = request_extractor(utterance)

        if requested is not None:
            next_dialog_state = 9

            system_utterance = response_generation("provide", restaurant=dialog_state["restaurant"], requested=requested)         

            
    elif current_state in [8, 9] and dialog_act == "confirm":
        
        slots = keyword_slot_extractor(utterance, current_state)

        for slot in ["food", "area", "pricerange"]:

            if slots[slot] is not None:
                system_utterance = response_generation("confirmproperty", restaurant=dialog_state["restaurant"], requested=slot, suggested_value=slots[slot])

    
    #additional requirements
    elif current_state == 10: 
        
        geen_additional = ["none", "nothing else"]
        
        if dialog_act in ["negate", "deny"] or utterance in geen_additional:
            next_dialog_state = 8                               
            system_utterance = response_generation("recommendrestaurant", restaurant=dialog_state["restaurant"])
            
        else:
            slots = keyword_slot_extractor(utterance, current_state)
      
            if slots["consequent"] is not None:
                dialog_state["consequent"] = slots["consequent"]
    
                recommendation, alternatives, contra = reasoning_component(dialog_state["alternatives"], dialog_state["consequent"])
    
                if recommendation is not None:
                    dialog_state["restaurant"] = recommendation
                    dialog_state["alternatives"] = alternatives
                    next_dialog_state = 8
                    system_utterance = response_generation("recommendrestaurant2", restaurant=recommendation, consequent=dialog_state["consequent"], contradiction=contra)
                else:
                    next_dialog_state = 12
                    system_utterance = response_generation("nomatch2",consequent=dialog_state["consequent"])
    
            else:
                system_utterance = response_generation("unknownpreference")
    
            
    if system_utterance is None and dialog_state["confirmation"] is None:

        if (dialog_state["food"] is not None and dialog_state["area"] is not None and dialog_state["pricerange"] is not None):

            recommendation, alternatives = restaurant_lookup(dialog_state["area"], dialog_state["food"], dialog_state["pricerange"])

            if not alternatives:
                if recommendation is not None:                   
                    dialog_state["restaurant"] = recommendation
                    dialog_state["alternatives"] = alternatives
        
                    next_dialog_state = 8                               
                    system_utterance = response_generation("recommendrestaurant", restaurant=recommendation)
        
                else:
                    next_dialog_state = 7                            
                    system_utterance = response_generation("nomatch")
            else:                
                    dialog_state["restaurant"] = recommendation
                    alternatives.append(recommendation)
                    dialog_state["alternatives"] = alternatives
                    next_dialog_state = 10
                    system_utterance = response_generation("askadditionalrequirements")
                

        elif dialog_state["food"] is None:
            next_dialog_state = 3                                    
            system_utterance = response_generation("askfoodtype")

        elif dialog_state["area"] is None:
            next_dialog_state = 4                                    
            system_utterance = response_generation("askarea")

        elif dialog_state["pricerange"] is None:
            next_dialog_state = 5                                    
            system_utterance = response_generation("askpricerange")
            
            
    if system_utterance is None:
        system_utterance = response_generation("null")
                
                
    dialog_state["state"] = next_dialog_state
    dialog_state["last_response"] = system_utterance
    
    return(dialog_state, system_utterance)
