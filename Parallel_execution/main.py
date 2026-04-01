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
    You need to rate the joke from 1 to 10.
    rank 1 is the worst joke, rank 10 is the best joke.
    """
    joke_idea: JokeModel = dspy.InputField(description="The joke idea")
    joke_rankings: list[int] = dspy.OutputField(description="Rating the joke from 1 to 10")



class ConditionalJokeGenerator(dspy.Module):
    def __init__(self, max_attempts: int = 3):
        self.query_to_idea = dspy.Predict(QueryToIdea)
        self.idea_to_joke = dspy.Predict(IdeaToJoke)
        self.joke_judge = dspy.ChainOfThought(JokeJudge)
        self.max_attempts = max_attempts

    async def async_forward(self, query: str):
        joke_ideas = await asyncio.gather(
            *[
                self.query_to_idea.acall(query=query),
                for _ in range(self.max_attempts)
            ]
        )
        print(f"Joke ideas: {joke_ideas}")

        judge_score = self.judge(joke_idea=joke_ideas).joke_rankings
        print(f"Judge score: {judge_score}")

        best_joke_idea_idx = judge_score.index(1)

        print(f"Best joke idea: {joke_ideas[best_joke_idea_idx]}")
        selected_joke_idea=joke_ideas[best_joke_idea_idx]
        print("Selected Joke Idea: \n", selected_joke_idea)

        joke = self.idea_to_joke(joke_idea=selected_joke_idea)

        return joke
        


async def main():
    joke_generator = ConditionalJokeGenerator()
    joke = await joke_generator.async_forward(query="Tell me a joke about a cat")
    print(f"Joke: {joke.joke}")

if __name__ == "__main__":
    asyncio.run(main())