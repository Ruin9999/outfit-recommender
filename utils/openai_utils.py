from openai import OpenAI
from pydantic import BaseModel
from typing import Optional

ASSISTANT_PROMPT = """
You are a personal stylist, I'll provide you with:
1) A detailed list of my clothing items (descriptions)
2) A description of what I want to achieve for the outfit.
Using only the items from my clothing items, propose an outfit that perfectly matches the requested vibe.
For each outfit:
  - Briefly explain why it suits the vibe.
  - Highlight any key details (e.g. color, pattern, texture) that I can use to improve the outfit.
  - Ensure all recommendations fit the mood or style I've described and reflect a cohesive, fashionable look.
Also give me an extremely brief description of the outfit.
You can add more details if you want. Always include a full outfit description from head to legs.

If I have no items in my wardrobe, come up with a creative outfit that fits the vibe.
"""

class RecommendationResponseFormat(BaseModel):
  explanation: str
  outfitDescription: str
  extraRecommendations: str

class OpenAIHelper():
  def __init__(self):
    self.client = OpenAI()

  def get_recommendation(self, rule_prompt: str, assistant_prompt: Optional[str] = None) -> RecommendationResponseFormat:
    recommendation = self.client.beta.chat.completions.parse(
      model="gpt-4o",
      response_format=RecommendationResponseFormat,
      messages=[
        { "role": "system", "content": assistant_prompt or ASSISTANT_PROMPT },
        { "role": "user", "content": rule_prompt }
      ]
    )
    return recommendation.choices[0].message.parsed #type: ignore