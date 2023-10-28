=======
Farming
=======

**Crops**

Many crops could be modeled, including wheat, barley, peas, grapes, figs, and
olives. The first crop to be modeled will be wheat.


**Research**

In Sumeria barley was sown in October/November. Fields were weeded throughout
the winter. Harvest was in April/May. Grain is stored by June/July at the latest.

**Sow and harvest cycle**

Crop will have a sowing date and a harvest date. Each crop will have two
modelled attributes while growing: progress to harvest and projected yield.

Assume we have equilibrium between wheat harvest and consumption at baseline
with mean temperature. If we have data suggesting that increases in temperature
above mean reduces yield and decreases below increase it, we can compute
variability. See figure 2 from Nature article to get marginal impact on yield of
each day's temperature.

High temperature causes crops to ripen faster, but I don't have a good idea of
by how much.

According to the springer paper, tracking cumulative rainfall from December -
February, rainfall from March - May, and consecutive dry days are the most
relevant rainfall metrics. Reasonable year-over-year variation could be as high
as 1200%. Rainfall can explain ~40% of variation.


**Land under cultivation**

Land under cultivation will be a function of population and arable land. The
amount of arable land is a function of terrain and irrigation. Land under
cultivation is proportional to the population of the village and capped by the
amount of arable land available.

**Amount of crop planted**

The amount of crop planted will be a function of land under cultivation. This
will have a corresponding "expected yield". This expected yield will be acted
on by various factors during the growing period to determine the actual yield.

**Impact of weather**

Very high temperatures even for a single day reduce yield significantly. Lower
temperatures increase yield slightly. Rainfall during the mid and late growing
seasons impact yield significantly. Having multiple dry days in a row reduces
yield, but by less than the cumulative impact of rainfall quantities.

**Impact of terrain**

**Impact of manpower**

**Impact of war**

**Impact of disasters**

- Pestilence
- Flood
- Drought

**Impact of irrigation**


**Sources**

- https://en.wikipedia.org/wiki/Agriculture_in_Mesopotamia
- https://www.worldhistory.org/article/9/agriculture-in-the-fertile-crescent--mesopotamia/
- https://archive.gci.org/articles/harvest-seasons-of-ancient-israel/
- https://www.nature.com/articles/s41467-020-18317-8/figures/2
- https://link.springer.com/article/10.1007/s10584-018-2170-x