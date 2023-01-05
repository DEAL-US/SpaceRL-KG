Training and Testing
=====================
The training and testing window is a class that is divided into 2 subframes which can be alternated.
It also sonatains some general functions that are shared between them.

The menu class holds several functions 
.. autoclass:: GUI.test_train_menu.menu

A generic entry point

.. autofunction:: GUI.test_train_menu.menu.add_elements

which calls the **train specific** builders:

.. autofunction:: GUI.test_train_menu.menu.add_train_elements
.. autofunction:: GUI.test_train_menu.menu.grid_trainframe

and the **test specific** builders:

.. autofunction:: GUI.test_train_menu.menu.add_test_elements
.. autofunction:: GUI.test_train_menu.menu.grid_testframe

Logic
------
these functions aid with adding and removing test and train elements and fill out dynamic elements.

.. autofunction:: GUI.test_train_menu.menu.add_to_list
.. autofunction:: GUI.test_train_menu.menu.remove_from_list
.. autofunction:: GUI.test_train_menu.menu.populate_embedding_listbox_test

Scrollables
-------------
These functions are meant to create an scrollable menu for the canvas which contains the tests and train elements.

.. autofunction:: GUI.test_train_menu.menu._bound_to_mousewheel
.. autofunction:: GUI.test_train_menu.menu._unbound_to_mousewheel
.. autofunction:: GUI.test_train_menu.menu._on_mousewheel
.. autofunction:: GUI.test_train_menu.menu._configure_window


Validation
-------------
Finally some helper validation functions for some of the elements.

.. autofunction:: GUI.test_train_menu.menu.ValidateRange
.. autofunction:: GUI.test_train_menu.menu.InvalidInput

