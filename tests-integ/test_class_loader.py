from strands.agent.agent import Agent
from strands.tools.class_loader import load_tools_from_instance


class WeatherTimeTool:
    def get_weather_in_paris(self) -> str:
        return f"sunny"

    @staticmethod
    def get_time_in_paris(r) -> str:
        return f"15:00"


def test_agent_weather_and_time():
    tool = WeatherTimeTool()
    tools = load_tools_from_instance(tool)
    prompt = (
        "What is the time and weather in paris?"
        "return only with the weather and time for example 'rainy 04:00'"
        "if you cannot respond with 'FAILED'"
    )
    agent = Agent(tools=tools)
    response = agent(prompt)
    text = str(response).lower()
    assert "sunny" in text.lower()
    assert "15:00" in text
