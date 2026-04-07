import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.rl_agent import RLAgent
from agents.greedy_agent import GreedyAgent
from env.environment import Ops

def test_rl_flow():
    print("\n--- RL Flow Verification ---")
    # Quick sanity: RL agent trains 10 eps
    agent = RLAgent(epsilon=0.5, seed=42)
    scores = agent.train(n_episodes=10, difficulty='medium', verbose=True)
    print('RL scores (small subset):', [round(s,3) for s in scores])
    print('Final weights:', [round(w,4) for w in agent._weights])
    
    # Save and reload
    os.makedirs('results', exist_ok=True)
    agent.save('results/rl_weights.json')
    agent2 = RLAgent.load('results/rl_weights.json')
    print('Loaded agent name:', agent2.name())

    # Quick inference
    env = Ops(difficulty='medium', seed=99)
    obs = env.reset()
    agent2.reset()
    done = False
    while not done:
        action = agent2.act(obs)
        obs, r, done, info = env.step(action)
    score = env.final_score()
    print(f'RL inference score: {score.total:.4f}')
    print("----------------------------\n")

if __name__ == "__main__":
    test_rl_flow()
