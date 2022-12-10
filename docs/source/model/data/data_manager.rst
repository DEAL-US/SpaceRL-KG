Data Manager 
=============
The data manager class is the one responsible for handling file creation, writing, saving and loading the agent models, the caches, the log and debug files and dataset information.

.. autoclass:: model.data.data_manager.DataManager

It's methods can be divided into 4 categories

Datasets & Caches
------------------
Handles loading datasets for training and testing as well as handling the required caching information

.. autofunction:: model.data.data_manager.DataManager.get_dataset
.. autofunction:: model.data.data_manager.DataManager.get_cache_for_dataset
.. autofunction:: model.data.data_manager.DataManager.save_cache_for_dataset

Agent & Models
---------------
Handles the saving and loading of agent models.

.. autofunction:: model.data.data_manager.DataManager.save_agent_model
.. autofunction:: model.data.data_manager.DataManager.restore_saved_agent
.. autofunction:: model.data.data_manager.DataManager.restore_saved_agent_PPO
.. autofunction:: model.data.data_manager.DataManager.saveall

Debug & Logs
--------------
Handles writing in the log files and the debug information files.

.. autofunction:: model.data.data_manager.DataManager.write_log
.. autofunction:: model.data.data_manager.DataManager.debug_save
.. autofunction:: model.data.data_manager.DataManager.debug_load

Miscellaneous
--------------
Miscellaneous operations.

.. autofunction:: model.data.data_manager.DataManager.update_lastest_input
.. autofunction:: model.data.data_manager.DataManager.copy_config
.. autofunction:: model.data.data_manager.DataManager.run_integrity_checks
.. autofunction:: model.data.data_manager.DataManager.remove_folders
