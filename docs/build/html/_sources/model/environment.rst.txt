Environment
===========
The environment class manages the Reinforcement Leaning environment.
It inherits from OpenAIs gym class which defines several core functions for an environment.

.. autoclass:: model.environment.KGEnv

It's functions can be divided into 4 sections

Gym
----
OpenAIs gym provides several functions, we override several of them to fit our environment, these are:

.. autofunction:: model.environment.KGEnv.step 
.. autofunction:: model.environment.KGEnv.reset 
.. autofunction:: model.environment.KGEnv.action_space 
.. autofunction:: model.environment.KGEnv.observation_space 

Caches
------
We build a reward cache to avoid calculating the same reward twice since it is a costly task.
These functions handle the initialization and storage of these caches.

.. autofunction:: model.environment.KGEnv.cache_init 
.. autofunction:: model.environment.KGEnv.save_current_cache 

States, Observations & Actions
-------------------------------
These functions manage the observations and actions that are currently available in the environment.
They calculate where the agent can and cannot go, how probable an action is and which triples are left to explore.

.. autofunction:: model.environment.KGEnv.get_current_state 
.. autofunction:: model.environment.KGEnv.get_encoded_state 
.. autofunction:: model.environment.KGEnv.get_encoded_observations 
.. autofunction:: model.environment.KGEnv.select_target 
.. autofunction:: model.environment.KGEnv.reset_queries 
.. autofunction:: model.environment.KGEnv.update_actions 

Rewards & Embeddings
---------------------
These functions manage the embeddings and rewards calculated for each step.

.. autofunction:: model.environment.KGEnv.get_distance 
.. autofunction:: model.environment.KGEnv.dist_func 
.. autofunction:: model.environment.KGEnv.calculate_embedding_min_max 
.. autofunction:: model.environment.KGEnv.get_embedding_info 

Pair dictionary
---------------
The caches use a special implementation of the dict class that can only ever accept tuples as keys.

.. autoclass:: model.environment.pairdict

These are the overriden function sigantures

.. autofunction:: model.environment.pairdict.__getitem__ 
.. autofunction:: model.environment.pairdict.__setitem__ 

and the auxiliary function 

.. autofunction:: model.environment.pairdict.tuple_check 
