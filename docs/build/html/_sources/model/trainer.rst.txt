Trainer
=========
The trainer class is the one responsible for initializing the training cycle, its one of the entry points to working with the module.

.. autoclass:: model.trainer.Trainer

The functions for the class are the following.

.. autofunction:: model.trainer.Trainer.run_prep
.. autofunction:: model.trainer.Trainer.run
.. autofunction:: model.trainer.Trainer.episode_misc
.. autofunction:: model.trainer.Trainer.debug_handle
.. autofunction:: model.trainer.Trainer.set_gpu_config
.. autofunction:: model.trainer.Trainer.run_debug
.. autofunction:: model.trainer.Trainer.update_gui_vars

The trainer also contains a GUI connector that handles GUI updates.

.. autoclass:: model.trainer.TrainerGUIconnector

Which contains the following methods.

.. autofunction:: model.trainer.TrainerGUIconnector.start_connection
.. autofunction:: model.trainer.TrainerGUIconnector.update_current_trainer
.. autofunction:: model.trainer.TrainerGUIconnector.update_info_variables
.. autofunction:: model.trainer.TrainerGUIconnector.threaded_update

Finally some miscellaneous methods as well as the main method which is the entry point.

.. autofunction:: model.trainer.get_gui_values
.. autofunction:: model.trainer.main