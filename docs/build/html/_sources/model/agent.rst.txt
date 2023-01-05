Agent
=========
The agent class is tasked with generating the Tensorflow models that will serve 
as the Reinforcement Learning agents which will carry out the path exploration.

.. autoclass:: model.agent.Agent

It can be separated into 4 sections.

Network Building
------------------
These set of functions generate the different sections of the models layers and
its intermediate components.

.. autofunction:: model.agent.Agent.policy_config 
.. autofunction:: model.agent.Agent.build_policy_networks 
.. autofunction:: model.agent.Agent.build_critic_network 
.. autofunction:: model.agent.Agent.build_network_from_copy 
.. autofunction:: model.agent.Agent.lstm_middle 

Actions & Rewards
------------------
These set of functions handles the generation of the reward values,
propagation of them to them to the network and the selection of actions.

.. autofunction:: model.agent.Agent.encode_action 
.. autofunction:: model.agent.Agent.get_next_state_reward 
.. autofunction:: model.agent.Agent.get_inputs_and_rewards 
.. autofunction:: model.agent.Agent.get_network_outputs 
.. autofunction:: model.agent.Agent.pick_action_from_outputs 
.. autofunction:: model.agent.Agent.select_action 
.. autofunction:: model.agent.Agent.select_action_runtime 

Policy Updates
------------------
These set of functions handle what the model recieves as update inputs.

.. autofunction:: model.agent.Agent.update_target_network 
.. autofunction:: model.agent.Agent.get_y_true 
.. autofunction:: model.agent.Agent.learn 
.. autofunction:: model.agent.Agent.calculate_backpropagation_rewards 
.. autofunction:: model.agent.Agent.calculate_PPO_rewards 

Auxiliary
------------------
Auxiliary functions.

.. autofunction:: model.agent.Agent.reset 
.. autofunction:: model.agent.Agent.remember 
.. autofunction:: model.agent.Agent.numpyfy_memories 
.. autofunction:: model.agent.Agent.stringify_actions 