import sys
import os
sys.path.insert(0, '.')
from agents.rl_agent import RLAgent
from env.environment import Ops

def test_benchmark():
    diffs = ['easy', 'medium', 'hard']
    seeds = [42, 123, 777]
    
    print(f"{'DIFFICULTY':<12} | {'SEED':<6} | {'SCORE':<8} | {'SLA':<8}")
    print("-" * 45)
    
    total_score = 0
    total_sla = 0
    count = 0
    
    for diff in diffs:
        for seed in seeds:
            env = Ops(difficulty=diff, seed=seed)
            obs = env.reset()
            agent = RLAgent.load('results/rl_weights.json')
            agent.epsilon = 0.0
            
            done = False
            while not done:
                obs, r, done, info = env.step(agent.act(obs))
            
            final = env.final_score().to_dict()
            total_score += final['total']
            total_sla += final['completion']
            count += 1
            
            print(f"{diff:<12} | {seed:<6} | {final['total']:.4f} | {final['completion']*100:.1f}%")
    
    print("-" * 45)
    print(f"{'OVERALL AVG':<12} | {' ':<6} | {total_score/count:.4f} | {total_sla/count*100:.1f}%")

if __name__ == "__main__":
    test_benchmark()
