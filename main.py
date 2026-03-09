from agent.core import Agent
from config.setting import LLM_PROVIDER


def main():
    print("=" * 50)
    print("  🤖 AI Agent — General Assistant")
    print(f"  Provider: {LLM_PROVIDER}")
    print("=" * 50)
    print("Commands: 'quit' to exit | 'reset' to clear history")
    print("          'switch openai' or 'switch anthropic' to change LLM")
    print("-" * 50)

    agent = Agent()

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            if user_input.lower() == "reset":
                agent.reset()
                print("Conversation cleared.")
                continue

            if user_input.lower().startswith("switch "):
                provider = user_input.split(" ", 1)[1].strip()
                agent.switch_provider(provider)
                continue

            response = agent.chat(user_input)
            print(f"\nAgent: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error] {str(e)}")


if __name__ == "__main__":
    main()