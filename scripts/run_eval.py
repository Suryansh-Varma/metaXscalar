import argparse
from evaluation.evaluator import evaluate_all


# ----------------------------
 
# ----------------------------
def get_agent(agent_type):
    if agent_type == "random":
        from agents.random_agent import RandomAgent
        return RandomAgent(seed=0)

    elif agent_type == "greedy":
        from agents.greedy_agent import GreedyAgent
        return GreedyAgent()

    elif agent_type == "rl":
        from agents.rl_agent import RLAgent
        import os
        weights = "results/rl_weights.json"
        if os.path.exists(weights):
            return RLAgent.load(weights)
        return RLAgent()

    else:
        raise ValueError("Unknown agent type")


# ----------------------------
# Main Entry
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="greedy",
                        choices=["random", "greedy", "llm", "rl"])
    parser.add_argument("--difficulties", nargs="+",
                        default=["easy", "medium", "hard"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 123, 777])
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Create agent safely
    agent = get_agent(args.agent)

    # Run evaluation
    evaluate_all(
        agent=agent,
        difficulties=args.difficulties,
        seeds=args.seeds,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()