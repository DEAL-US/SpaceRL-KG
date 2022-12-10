Main Menu
=========
The main menu class contains the elements to render the main window and the logic to connect to the rest of the module.

.. autoclass:: GUI.main.mainmenu

We can separate the functions into 7 different categories

Window Structure
----------------
Responsible for the correct position and size of different Gui elements.

.. autofunction:: GUI.main.mainmenu.add_elements
.. autofunction:: GUI.main.mainmenu.grid_elements
.. autofunction:: GUI.main.mainmenu.add_styles

Submenu Handling
----------------
Responsible for creating and managing information from submenus

.. autofunction:: GUI.main.mainmenu.open_menu
.. autofunction:: GUI.main.mainmenu.extract_config_on_close
.. autofunction:: GUI.main.mainmenu.extract_info_on_close
.. autofunction:: GUI.main.mainmenu.pathmenu_teardown

Execution
----------------
These functions handle the execution of the model module.

.. autofunction:: GUI.main.mainmenu.run_experimentation
.. autofunction:: GUI.main.mainmenu.run_tests


Thread Handling
----------------
Launching threads and mantaining their states.

.. autofunction:: GUI.main.mainmenu.kill_all_threads
.. autofunction:: GUI.main.mainmenu.mainthread

Error Handling
----------------
Visual updates to report errors to the user.

.. autofunction:: GUI.main.mainmenu.showbusyerror
.. autofunction:: GUI.main.mainmenu.change_error_text

Updates
----------------
Handles the updating of the different parts of the GUI and program

.. autofunction:: GUI.main.mainmenu.update_connectors
.. autofunction:: GUI.main.mainmenu.update_progress
.. autofunction:: GUI.main.mainmenu.launch_updater
.. autofunction:: GUI.main.mainmenu.change_progtext
.. autofunction:: GUI.main.mainmenu.set_progbar_value
.. autofunction:: GUI.main.mainmenu.change_progbar_max
.. autofunction:: GUI.main.mainmenu.update_all_progress


Miscellaneous
----------------
Miscellaneous helper functions.

.. autofunction:: GUI.main.mainmenu.open_folder
.. autofunction:: GUI.main.mainmenu.intercept_close
