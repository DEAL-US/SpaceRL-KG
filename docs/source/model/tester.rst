Tester
=========
The tester class is the one responsible for initializing the testing cycle, its one of the entry points to working with the module.

.. autoclass:: model.tester.Tester

The functions for the class are the following.

.. autofunction:: model.tester.Tester.run
.. autofunction:: model.tester.Tester.generate_MRR_boxplot_and_source
.. autofunction:: model.tester.Tester.generate_found_paths_files
.. autofunction:: model.tester.Tester.path_contains_entity
.. autofunction:: model.tester.Tester.set_gpu_config
.. autofunction:: model.tester.Tester.update_gui_vars

The tester also contains a GUI connector that handles GUI updates.

.. autoclass:: model.tester.TesterGUIconnector

Which contains the following methods.

.. autofunction:: model.tester.TesterGUIconnector.start_connection
.. autofunction:: model.tester.TesterGUIconnector.update_current_tester
.. autofunction:: model.tester.TesterGUIconnector.update_info_variables
.. autofunction:: model.tester.TesterGUIconnector.threaded_update

There are also some miscellaneous methods.

.. autofunction:: model.tester.compute_metrics
.. autofunction:: model.tester.get_agents
.. autofunction:: model.tester.extract_config_info
.. autofunction:: model.tester.get_gui_values

and the main method which starts the workflow.

.. autofunction:: model.tester.main
