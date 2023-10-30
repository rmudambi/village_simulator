class Pipelines:
    # Weather pipelines
    TEMPERATURE: str = "temperature"
    RAINFALL: str = "rainfall"

    # Demographics pipelines
    FERTILITY_RATE = "fertility_rate"
    MORTALITY_RATE = "mortality_rate"
    TOTAL_POPULATION = "total_population"

    # Resource pipelines
    @staticmethod
    def get_resource_consumption(resource: str) -> str:
        return f"{resource}.consumption"

    @staticmethod
    def get_resource_accumulation(resource: str) -> str:
        return f"{resource}.accumulation"

    # Wheat pipelines
    PROJECTED_WHEAT_HARVEST: str = "wheat.projected_harvest"
