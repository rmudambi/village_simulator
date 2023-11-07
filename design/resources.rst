=================
General Resources
=================

**Resources**

Many resources could be modeled, including stone, wood, copper, tin, iron, gold,
and silver. The first resources to be modeled will be wood and stone.


**Research**

TBD

**Basic design**

Resources are modeled by using two columns - one to represent the amount of
potential resources available to be extracted (i.e. stone in the ground or wood
in trees), and one to represent the amount that has been extracted and is stored,
ready to be used.

Tiles are initialized with a certain amount of potential
resources which is randomly determined and dependent on the terrain of the tile.
Depending on the resource, it may be a renewable resource (e.g. wood) or a
non-renewable resource (e.g. stone).

On each time-step, the amount of resources extracted is determined by the
amount of manpower allocated to the task, the amount of resources available,
and the amount of resources that can be extracted by the manpower. Initially,
manpower allocated to extraction will be modeled using a total population as a
proxy.

On each time-step, the amount of resources consumed is determined by the
amount of resources available and the population. Later on, specific uses of the
resource in question may be used in addition, when calculating consumption.

On each time-step, regeneration of renewable resources is determined by the
using a resource-specific regeneration function. This function will be
determined by the type of resource and the terrain of the tile.


**Sources**

TBD