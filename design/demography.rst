==========
Demography
==========

The document described how population dynamics are modeled for each village. Each of the modeled attributes are
defined for each compartment.

**Compartmental Model**

- Sex:
    - Male
    - Female

**Mortality**

Each compartment has a mortality rate that is initialized as 40 deaths per 1000 person years.

*Effect of food consumption*

Mortality is directly correlated with the difference between "natural" food consumption and actual food consumption.
This is modeled by this relationship:

```
mortality ~ natural_food_consumption / food_consumption)
```

**Fertility**

Each compartment has a fertility rate that is initialized as 50 births per 1000 female person years.

*Seasonality*

Fertility has an annual seasonality that is defined by a sinusoidal function. The amplitude of the function is
defined by the `fertility_amplitude` parameter. Fertility peaks at a 6 month offset from the peak of the major
harvest season.

Note: `fertility_amplitude` has a stand-in value of 0.1. This value is not based on any empirical data.

*Effect of harvest*

Fertility is directly correlated with the magnitude of the previous harvest. A marginal change of 1 unit in the
previous harvest will result in a marginal change of 0.38 units in fertility.

**Sources**

- https://link.springer.com/article/10.1007/s11698-016-0144-7