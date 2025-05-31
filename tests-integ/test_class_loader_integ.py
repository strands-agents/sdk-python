from strands.agent.agent import Agent
from strands.tools.class_loader import load_tools_from_instance

class TestClassLoaderIntegration:
    class WeatherTimeTool:
        def get_weather(self, location: str) -> str:
            return f"The weather in {location} is sunny."
        @staticmethod
        def get_time(location: str) -> str:
            return f"The time in {location} is 2:00 PM."

    def test_agent_weather_and_time(self):
        tool = self.WeatherTimeTool()
        tools = load_tools_from_instance(tool, prefix="weather")
        agent = Agent(tools=tools)
        response = agent("What is the weather and time in Paris?")
        text = str(response).lower()
        assert "weather" in text or "sunny" in text
        assert "time" in text or "2:00" in text 