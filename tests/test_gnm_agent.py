import os, sys
# Compute the absolute path to the parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
parent_dir = os.path.dirname(script_dir)  # Parent directory
sys.path.insert(0, parent_dir)

from auto_agent import Agent_GNM
agent = Agent_GNM(f"{parent_dir}/auto_agents/gnm/config/models.yaml",f"{parent_dir}/auto_agents/gnm/weights/","gnm_large",f"{parent_dir}/out/maps/skokloster-castle_20231207133858358375/")

dist, wayp = agent.predict_currHistAndGoal(agent.topomap[:6],agent.topomap[7])
print(dist, wayp)