============
Weather v1.0
============

Overall, weather is modelled as regionally consistent acroess the entire
game-board. This will likely change in later versions of the game as the game
board becomes larger. The basic modelling strategy is to calculate on each
time-step "global" parameters to define for the weather feature on each
time-step. These values will be modified by a functional model of seasonality
and then used to draw a "global" value for that weather feature on the
time-step. Then there are addition parameters defining a distribution from which
to sample the the specific values for each individual tile.

**Seasons**

Seasons are modeled as annually recurring sinusoidal functions. These functions
provide values corresponding to the seasonal impact on weather generation
parameters. The impact calculated for a given day of the year can then be
applied additively or multiplicatively to the parameter in question.

**Temperature**

Temperature is modeled by using a normal distribution `d_1``to generate a global
mean value. That mean value is then used by another normal distribution `d_2` to
sample the temperature value for each tile of the map.

The standard deviations for the above defined distributions `d_1` and `d_2` are:

- `d_1`: 5.0
- `d_2`: 2.0

The mean value used to calculate the global mean value is modified by
seasonality. It can be computed using a sinusoidal function with the following
properties:

- Minimum temperature (in Fahrenheit): 50
- Maximum temperature (in Fahrenheit): 80
- Date of minimum temperature: January 15th
- Period: 1 year

**Rainfall**

Implemented by not yet documented

**Natural disasters**

- Pestilence
- Flood
- Drought

**Impact of terrain**

Initial version of the model will have uniform weather patterns, but it would be
interesting to introduce wet and dry zones, regional weather patterns, and the
effect of elevation and mountains on temperature and rainfall.

**Climate change**

If this game ends up modeling longer stretches of time, we could introduce
climate changes. This could mirror the actual changes in temperature and
rainfall

**Limitations**

- There is a global correlation of temperature and rainfall, but no mechanism to
  ensure that tiles closer to each other have more similar weather than those
  further away.
- Only temperature and rainfall are modelled, excluding other phenomena such as:
    - Wind direction and strength
    - Cloud cover/sunshine
    - Snow
    - Freezing/frost


**Sources**

-