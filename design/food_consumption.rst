================
Food Consumption
================

Food is consumed daily by the population. The amount of food consumed
is based on the population size and the amount of food available.  As
the stores of food are depleted, the amount of food consumed will
decrease. This will also result in an increase in the mortality rate.


**Food consumption** is calculated as follows:

`natural_rate` is the rate of food consumption given unconstrained food
availability.

`food_available` is the amount of food available in the village.

`population` is the current population of the village.

`time_to_next_harvest` is the number of days until the next harvest.

```
food_consumed = min(food_available, natural_rate * population, food_available / time_to_next_harvest)
```
