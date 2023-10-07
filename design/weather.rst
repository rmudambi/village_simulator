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
to sample the the specific values for each individual tile. The process can be
thought of like this:

`value(time, tile) = value(global_value(time), tile)`

Note: all numeric values specified in this document are configurable, and might
be changed in the future causing the document to be out of sync with the true
values. This document should be modified to reflect any methodological changes
in the v1.0 model.

**Seasons**

Seasons are modeled as annually recurring sinusoidal functions. These functions
produce quantities that describe various seasonal impacts. These seasonal
impacts then have concrete effects on the calculation of both temperature and
rainfall. All seasonal effects operate on an annual cycle.

*Temperature*

The `seasonal_temperature_shift` is computed using a sinusoidal function with
`amplitude = 15` and `minimum_date = January 15th`

*Rainfall*

An `aridity_factor` is computed using a sinusoidal function with `min = 0.1`,
`max = 1.0`, and `minimum_date = August 15th`

**Temperature**

Temperature is measured in degrees Fahrenheit.

It is modeled by producing the global mean temperature at a given time and then
sampling specific values for each tile.

`global_temperature(time)` samples from a normal distribution with
`mean = 65.0 + seasonal_temperature_shift` and `standard deviation = 5.0`.
`seasonal_temperature_shift` is described above in the Seasons section.

`temperature(time, tile)` samples from a normal distribution with
`mean = global_temperature(time)` and `standard deviation = 5.0`.

**Rainfall**

Rainfall is measured in mm per day.

It is modeled by first determining whether it rains at all that day. If it
rains, the simulation produces a global mean rainfall amount and then samples
specific values for each tile. The process can be thought of like this:

`no_rain(time)` samples from a dichotomous distribution with
`probability_no_rain = 0.55 * aridity_factor`.

`global_rainfall(time)` is 0.0 if `no_rain(time)`, otherwise it samples from a
gamma distribution with `shape = 0.9902` and `scale = 15.0 * aridity_factor`.

`rainfall(time, tile)` remains 0.0 if `no_rain(time)`, but samples from a normal
distribution truncated at 0 (as it's not possible to have negative rainfall).
This truncated normal distribution has `mean = global_rainfall(time)` and
`standard deviation = 0.05 * global_rainfall(time)`.

The `aridity_factor` is described above in the Seasons section.

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