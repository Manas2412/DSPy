from print_utils import print
import dspy
from typing import Optional
from pydantic import BaseModel, Field

dspy.configure(lm = dspy.LM("ollama/llama3.1:8b"))


class JokeModel(BaseModel):
    setup: str
    contradiction: str
    punchline: str

class QueryToIdea(dspy.Signature):
    """
    You are a helpful assistant that converts a user query into a list of ideas.
    """
    query: str = dspy.InputField(description="The user query")
    ideas: list[str] = dspy.OutputField(description="The list of ideas")

class IdeaToJoke(dspy.Signature):
    """
    You are a comedian who likes to tell stories about your life before delivering a punchline.
    """
    ideas: JokeModel = dspy.InputField(description="The list of ideas")
    joke: JokeModel = dspy.OutputField(description="The joke")

class JokeJudge(dspy.Signature):
    """
    You are a judge of jokes. You are given a joke and you need to decide if it is funny.
    """
    joke_idea: JokeModel = dspy.InputField(description="The joke idea")
    joke_rating: int = dspy.OutputField(description="Rating the joke from 1 to 10")



class ConditionalJokeGenerator(dspy.Module):
    def __init__(self, max_attempts: int = 3):
        self.query_to_idea = dspy.Predict(QueryToIdea)
        self.idea_to_joke = dspy.Predict(IdeaToJoke)
        self.joke_judge = dspy.ChainOfThought(JokeJudge)
        self.max_attempts = max_attempts

    def run(self, query: str) -> JokeModel:
        for _ in range(self.max_attempts):
            print(f"Attempt {_ + 1} of {self.max_attempts}")
            joke_idea = self.query_to_idea.run(query=query)
            print(f"Joke ideas: {joke_idea.ideas}")
            joke = self.idea_to_joke.run(ideas=joke_idea.ideas)
            print(f"Joke: {joke.joke}")
            joke_rating = self.joke_judge.run(joke_idea=joke_idea, joke=joke.joke)
            print(f"Joke rating: {joke_rating.joke_rating}")
            if joke_rating.joke_rating >= 7:
                return joke
            else:
                print(f"Joke is not funny. Trying again...")
                query = f"Please refine the joke: {joke.joke}"
        return None

joke_generator = JokeGenerator()
joke = joke_generator.run(query="Tell me a joke about a cat")
print(f"Joke: {joke.joke}")