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
    draft_joke: Optional[str] = dspy.InputField(description="The draft joke")
    feedback: Optional[str] = dspy.InputField(description="The feedback on the draft joke")
    joke: JokeModel = dspy.OutputField(description="The joke")

class Refinement(dspy.Signature):
    """
    Given a joke, is it funny? If not, suggest a change.
    """
    joke_id: str = dspy.InputField(description="The joke id")
    joke: str = dspy.InputField(description="The joke")
    changes: str = dspy.OutputField(description="The changes to the joke")

class JokeGenerator(dspy.Module):
    def __init__(self, n_attempts: int = 3):
        self.query_to_idea = dspy.Predict(QueryToIdea)
        self.idea_to_joke = dspy.Predict(IdeaToJoke)
        self.refinement = dspy.ChainOfThought(Refinement)
        self.n_attempts = n_attempts

    def run(self, query: str) -> JokeModel:
        joke_idea = self.query_to_idea.run(query=query)
        print(f"Joke ideas: {joke_idea.ideas}")

        draft_joke = None
        feedback = None

        for _ in range(self.n_attempts):
            joke = self.idea_to_joke.run(ideas=joke_idea.ideas, draft_joke=draft_joke, feedback=feedback)
            print(f"Joke: {joke.joke}")
            feedback = self.refinement.run(joke_id=joke.joke_id, joke=joke.joke)
            print(f"Feedback: {feedback.changes}")
            draft_joke = joke.joke

        return joke

joke_generator = JokeGenerator()
joke = joke_generator.run(query="Tell me a joke about a cat")
print(f"Joke: {joke.joke}")